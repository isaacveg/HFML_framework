from math import ceil
import os

os.environ['CUDA_VISIBLE_DEVICES']='7'
os.environ['CUDA_LAUNCH_BLOCKING']='1'

# Set cuda information before importing FLamingo
import sys
sys.path.append('../FLamingo/')
from FLamingo.core.server import *
from models import create_model_instance_custom
import numpy as np


class MetaClientInfo(ClientInfo):
    def __init__(self, rank, psi):
        # 接受的信息
        # - 训练前直接测试准确率和损失，训练前微调后测试准确率和损失，训练后直接测试准确率和损失，训练后微调后测试准确率和损失
        # - 训练后的模型参数，训练过程中的v_info，训练过程中的样本，训练过程中的损失
        super().__init__(rank)
        self.bf_test_acc = 0.0
        self.bf_test_loss = 0.0
        self.bf_optim_loss = 0.0
        self.bf_optim_samples = 0
        self.bf_test_optim_acc = 0.0
        self.bf_test_optim_loss = 0.0
        self.train_loss = 0.0
        self.af_test_acc = 0.0
        self.af_test_loss = 0.0
        self.optim_loss = 0.0
        self.optim_samples = 0
        self.test_acc = 0.0
        self.test_loss = 0.0
        self.test_samples = 0
        # total dataset size
        self.full_batches = 0
        # V-info
        self.v_info = 0.0
        # self.V = 1.0
        # self.psi = psi
        self.train_samples = 0
        self.steps = 5
        self.outer_updates = 6
        self.tau = None     # Initially set to None
        # system info
        self.train_time = 0.0
        self.single_step_time = 0
        self.send_time = 0.0
        self.num_batches = 0
        # config info
        self.computation = 0.0
        self.dynamics = 0.0
        self.communication = 0.0
        # rolling mean
        self.send_time_list = []
        self.single_step_time_list = []
        self.v_info_list = [0.0]
        # self.V_list = [1.0]
        
    def update_rolling_mean_list(self):
        self.send_time_list.append(self.send_time)
        self.single_step_time_list.append(self.single_step_time)
        self.v_info_list.append(self.v_info)
        # # V is exp of psi * v_info, psi is temperate parameter
        # self.V = np.exp(self.psi * self.v_info)
        # self.V_list.append(self.V)
        if len(self.send_time_list) > 5:
            self.send_time_list.pop(0)
            self.single_step_time_list.pop(0)
            self.v_info_list.pop(0)
            # self.V_list.pop(0)
        self.send_time = np.mean(self.send_time_list)
        self.single_step_time = np.mean(self.single_step_time_list)
        self.v_info = np.mean(self.v_info_list)
        # self.V = np.mean(self.V_list)


class MetaServer(Server):
    def init(self):
        """
        Defining model and related information
        """
        self.network = NetworkHandler()
        self.model = create_model_instance_custom(self.model_type, self.dataset_type)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.993)
        
    # def aggregate(self, client_list=None, weight_by='avg'):
    def aggregate(self, client_list=None):
        """Aggregate params by weighted average.
        Args:
            client_list: list of client ranks
            weight_by: 'average', 'dataset_size', 'v_info'
        """
        client_list = client_list or self.selected_clients_idxes
        # if weight_by == 'avg':
        weight = 1/len(client_list)
        for rank in client_list:
            self.get_client_by_rank(rank).weight = weight
        # elif weight_by == 'upd':
        # if not self.ABLATION_V:
        #     total_size = sum(self.get_clients_attr_tolist('outer_updates', client_list))
        #     for rank in client_list:
        #         self.get_client_by_rank(rank).weight = self.get_client_by_rank(rank).outer_updates/total_size
        # else:
        #     for rank in client_list:
        #         self.get_client_by_rank(rank).weight = 1/len(client_list)
        self.log(f'Weighted average weight: {self.get_clients_attr_tolist("weight", client_list)}')
        self.weighted_average()

    def dml_decide(self, client_list=None):
        """
        Decide tau, steps, outer updates, and lr according to V, data volume, and time.
        """
        client_list = client_list or self.selected_clients_idxes
        # Calculate combined metric considering V, num_batches, and time
        combined_metric = {}
        for rank in client_list:
            client = self.get_client_by_rank(rank)
            # if self.trial == 'all':
            #     # Trial 1: client with long average time, harder samples are given lower metric
            #     total_time = client.full_batches * client.single_step_time + 2 * client.send_time
            #     combined_metric[rank] = (client.V * client.full_batches) * total_time  # Higher metric should prioritize client for lower tau
            # elif self.trial == 'two':
            #     # Trial 2: do not consider time when assigning inner updates, only V and Dataset Size
            #     combined_metric[rank] = client.V * client.full_batches    # Easier & higher info
            # elif self.trial == 'one':
            # Trial 3: only consider V to rank clients
            # combined_metric[rank] = client.V
            combined_metric[rank] = client.v_info
        # Sort clients based on combined metric in descending order
        sorted_clients = sorted(combined_metric.keys(), key=lambda rank: combined_metric[rank], reverse=True)

        # Set steps for sorted clients based on their rank
        steps_range = list(range(self.min_steps, self.max_steps + 1))
        num_steps = len(steps_range)
        clients_per_step = len(sorted_clients) // num_steps
        remainder_clients = len(sorted_clients) % num_steps
        
        current_index = 0

        for step in steps_range:
            for _ in range(clients_per_step):
                if current_index < len(sorted_clients):
                    client = self.get_client_by_rank(sorted_clients[current_index])
                    client.steps = step
                    current_index += 1
            # Distribute the remainder clients
            if remainder_clients > 0:
                client = self.get_client_by_rank(sorted_clients[current_index])
                client.steps = step
                current_index += 1
                remainder_clients -= 1

        if not self.ABL_ONE:
            # Set tau for each client based on their metric
            top_client = self.get_client_by_rank(sorted_clients[0])
            # top_client_tau = top_client.full_batches    # Highest metric given largest tau for thorough training
            # top_client.tau = max(min(top_client_tau, self.max_tau), self.min_tau)   # clip tau
            top_client_tau = self.mid_tau
            top_client.tau = top_client_tau
            # Adjust tau for remaining clients based on top client's estimated training time
            top_client_trainingtime = top_client.tau * top_client.single_step_time + 2 * top_client.send_time
                
            for rank in sorted_clients[1:]:
                client = self.get_client_by_rank(rank)
                estimated_training_time = top_client_trainingtime - 2 * client.send_time
                if estimated_training_time <= 0:
                    # the client is very slow, need adjustment
                    client.tau = self.min_tau
                else:
                    client_tau = int(estimated_training_time / client.single_step_time)
                    client.tau = max(min(client_tau, self.max_tau), self.min_tau)

            # Ensure all clients' tau are within the specified range
            max_time = -1.0
            for rank in sorted_clients:
                client = self.get_client_by_rank(rank)
                if client.tau * client.single_step_time + 2 * client.send_time > max_time:
                    max_time = client.tau * client.single_step_time + 2 * client.send_time
            for rank in sorted_clients:
                client = self.get_client_by_rank(rank)
                client_tau = int((max_time - 2 * client.send_time) / client.single_step_time)
                client.tau = max(min(client_tau, self.max_tau), self.min_tau)
                client.outer_updates = ceil(client.tau / client.steps)
        else:
            # set all tau to self.mid_tau
            for rank in sorted_clients:
                client = self.get_client_by_rank(rank)
                client.tau = self.mid_tau
                # client.outer_updates = ceil(client.tau / client.steps)
                client.outer_updates = 10
        # Update learning rates
        self.beta = self.reduce_lr(self.beta)

        # Log client information
        self.log(f'selected_clients_list: {client_list}')
        self.log(f'v_info: {self.get_clients_attr_tolist("v_info", client_list)}')
        # self.log(f'exp_v_info: {self.get_clients_attr_tolist("V", client_list)}')
        self.log(f'tau: {self.get_clients_attr_tolist("tau", client_list)}')
        self.log(f'steps: {self.get_clients_attr_tolist("steps", client_list)}')
        self.log(f'full_batches: {self.get_clients_attr_tolist("full_batches", client_list)}')
        self.log(f'single_step_time: {self.get_clients_attr_tolist("single_step_time", client_list)}')
        self.log(f'send_time: {self.get_clients_attr_tolist("send_time", client_list)}')
        self.log(f'outer_updates: {self.get_clients_attr_tolist("outer_updates", client_list)}')
        
    def dml_ablation(self, client_list=None):
        """
        Ablation study: ablation V to fix it
        """
        client_list = client_list or self.selected_clients_idxes
        for rank in client_list:
            client = self.get_client_by_rank(rank)
            client.tau = self.mid_tau
            client.steps = self.ablation_steps
            client.outer_updates = int(client.tau / client.steps)

        
    def reduce_lr(self, lr):
        return max(lr * self.lr_decay, 0.01)
    
    def gather_initial_info(self):
        """Our algorithm contains a pretraining phase, we can collect these information
        Need to update_rolling_mean_list
        """
        self.listen(src_ranks=self.all_clients_idxes)
        for client in self.all_clients:
            # Update rolling mean lists
            client.update_rolling_mean_list()
        self.log("Gathered initial info: computation, communication, dynamics, full_batches")
        self.log(f"{self.get_clients_attr_tolist('computation', self.all_clients_idxes)}")
        self.log(f"{self.get_clients_attr_tolist('communication', self.all_clients_idxes)}")
        self.log(f"{self.get_clients_attr_tolist('dynamics', self.all_clients_idxes)}")
        self.log(f"{self.get_clients_attr_tolist('full_batches', self.all_clients_idxes)}")
        self.log('='*10+'End of initial info'+'='*10)
            
    def run(self):
        self.init_clients(clientObj=MetaClientInfo, ex_args=[self.psi])
        if self.dataset_type == 'cifar10':
            self.generate_global_test_set()
        self.gather_initial_info()
        while True:
            """
            Server acts:
            1. Select clients, calculate training settings, send
            2. Listen to V-info and g_i
            3. Calculate aggregation weight and aggregate
            4. Decide steps, outer updates, and lrs
            5. Update info stored on server
            """
            if self.dataset_type == 'cifar10':
                global_test_dic = self.test(self.model, self.test_loader)
                self.log(f"global_acc: {global_test_dic['test_acc']}, global_loss: {global_test_dic['test_loss']}")
            # 1. Random select
            self.select_clients()
            # 2. Calculate tau st up hyper_params.
            if not self.ABLATION_V:
                self.dml_decide()
            else:
                # abl 
                self.dml_ablation()
            self.personalized_broadcast(
                common_data={
                    'status': 'TRAINING',
                    'params': self.export_model_parameter()
                },
                personalized_attr=['steps','outer_updates']
            )
            self.listen()
            # update big V
            for client in self.selected_clients:
                client.update_rolling_mean_list()
            # self.aggregate(weight_by=self.weight_by)
            self.aggregate()
            # log information
            dic = self.average_client_info(self.selected_clients_idxes,attrs=[
                'train_time',
                'bf_test_acc',
                'bf_test_loss',
                'bf_optim_loss',
                'bf_optim_samples',
                'bf_test_optim_acc',
                'bf_test_optim_loss',
                'train_loss',
                'train_samples',
                'v_info',
                'af_test_acc',
                'af_test_loss',
                'optim_loss',
                'optim_samples',
                'test_acc',
                'test_loss',
                'test_samples',
                'num_batches',
                'single_step_time'
            ])
            self.evaluate_on_new_clients()
            eval_dic = self.average_client_info(
                self.eval_clients_idxes, 
                attrs=['bf_test_acc', 'bf_test_loss', 'bf_test_optim_acc', 'bf_test_optim_loss'],
                dic_prefix='eval')
            self.finalize_round()
            self.quick_rec_dict(dic)
            self.quick_rec_dict(eval_dic)
            if self.global_round >= self.max_epochs:
                self.log(f'Reaching epochs limit {self.max_epochs}')
                break
        # out of loop
        self.log("Server stopped")
        self.stop_all()
        self.summarize()


if __name__ == "__main__": 
    server = MetaServer()
    server.run()