import os
from mpi4py import MPI

# from FLamingo.core.utils.model_utils import create_model_instance
WRLD = MPI.COMM_WORLD
RANK = WRLD.Get_rank()
os.environ['CUDA_VISIBLE_DEVICES'] = str(RANK%8)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# import FLamingo after setting cuda information
import sys
sys.path.append('../FLamingo/')

from FLamingo.core.client import *
from models import create_model_instance_custom
from FLamingo.core.utils.train_test_utils import infinite_dataloader
from v_info_utils import InfoDataset
from torch.utils.data import DataLoader


class MetaClient(Client):
    """
    Your own Client
    """
    def init(self):
        # self.model = AlexNet()
        self.model = create_model_instance_custom(self.model_type, self.dataset_type)
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        self.loss_func = torch.nn.CrossEntropyLoss()
        lambda_lr = lambda step: max(0.005, self.lr_decay ** step)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_lr)

    def rand_send(self):
        if self.model_type == 'alexnet':
            return self.rand_time(self.communication, self.dynamics)*14.6
        elif self.model_type == 'fedavgcnn':
            return self.rand_time(self.communication, self.dynamics)*2.3
        else:
            return self.rand_time(self.communication, self.dynamics)*3.1
    
    def rand_comp(self):
        if self.model_type == 'alexnet':
            return self.rand_time(self.computation, self.dynamics)
        elif self.model_type == 'fedavgcnn':
            return self.rand_time(self.computation, self.dynamics)
        else:
            return self.rand_time(self.computation, self.dynamics)

    def reptile_train(self, model, optimizer, loss_func, beta, outer_updates, steps, infoloader):
            """
            Reptile training using given functions.
            The main difference here is that this function will use a 
            different type of dataloader, i.e. info_dataloader, which will
            not only returns data, label, but also point-wise information
            for you to utilize.
            
            """
            model.train()
            model = model.to(self.device)
            total_loss = 0.0
            total_samples = 0
            train_time = 0.0
            total_v_info = 0.0
            s_t = time.time()
            
            inf_loader = infinite_dataloader(infoloader)
            
            for idx in range(outer_updates):
                original_model_vector = self.export_model_parameter(model)
                for st in range(steps):
                    data, label, pvis = next(inf_loader)
                    batch_num, batch_loss = self._train_one_batch(model, data.to(self.device), label.to(self.device), optimizer, loss_func)
                    total_samples += batch_num
                    total_loss += batch_loss * batch_num
                    # train_time += self.rand_time(self.computation, self.dynamics)
                    train_time += self.rand_comp()
                    total_v_info += sum(pvis.tolist())
                new_vec = self.export_model_parameter(model)
                updated_vector = self.update_model(original_model_vector, new_vec, beta)
                self.set_model_parameter(updated_vector, model)
            train_time = train_time if self.USE_SIM_SYSHET else time.time() - s_t 
            avg_loss = total_loss / total_samples
            avg_v_info = total_v_info / total_samples
            return {'train_loss':avg_loss, 'train_samples':total_samples, 'train_time':train_time, 'v_info': avg_v_info}

    def local_optimization(self, model, dataloader, optim_steps=None):
        """
        Local optimization for meta-model.
        
        Args:
            - model: model to be optimized
            - dataloader: dataloader for training
            - optim_steps: number of steps to train
        
        Returns:
            - dict: containing optim_loss, optim_samples, optim_time
        """
        if optim_steps is None:
            optim_steps = len(dataloader)
        model.train()
        model = model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        loss_func = torch.nn.CrossEntropyLoss()
        num, loss = 0, 0.0
        s_t = time.time()
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            batch_num, batch_loss = self._train_one_batch(model, data, target, optimizer, loss_func)
            num += batch_num
            loss += batch_loss * batch_num
            if idx+1 >= optim_steps:
                break
        optim_time = time.time() - s_t
        loss /= num
        return {'optim_loss':loss, 'optim_samples':num, 'optim_time':optim_time}
    
    def update_model(self, original, after, lr):
        """
        Update original model according to trained model
        """
        delta = after - original
        updated = original + lr * delta
        return updated
    
    def run(self):
        """
        Client jobs, usually have a loop
        """
        # DATA: send time, train time, and batch number
        # temp_send, temp_single = self.rand_time(self.communication, self.dynamics)*10, self.rand_time(self.computation, self.dynamics)
        temp_send, temp_single = self.rand_send(), self.rand_comp()
        self.send(data={
            'send_time': temp_send,
            'train_time': temp_single*len(self.train_loader),
            'full_batches': len(self.train_loader),
            'num_batches': len(self.train_loader),
            'train_samples': len(self.train_loader.dataset), 
            'single_step_time': temp_single,
            'computation': self.computation,
            'communication': self.communication,
            'dynamics': self.dynamics
        })
        # 获得数据集中所有数据的 pvi 信息
        dataset_with_info = InfoDataset(os.path.join(self.data_dir, f'info/{self.rank}.npz'))
        self.info_train_loader = DataLoader(dataset_with_info, batch_size=self.batch_size, shuffle=True)
        while True:
            # get from server
            data = self.listen()
            if data['status'] == 'TRAINING':
                self.set_model_parameter(data['params'])
                test_server = self.test(self.model, self.test_loader)
                # Using Reptile to train
                self.steps = data['steps']
                self.outer_updates = data['outer_updates']
                self.log(f'alpha: {self.alpha}')
                self.log(f'outer_updates: {self.outer_updates}')
                params = self.export_model_parameter()
                optim_server = self.local_optimization(self.model, self.train_loader, self.optim_steps)
                test_server_optim = self.test(self.model, self.test_loader)
                self.set_model_parameter(params)
                self.log(f"test_server: {test_server}")
                self.log(f"optim_server: {optim_server}")
                self.log(f"test_server_optim: {test_server_optim}")
                # Using Reptile to train
                info_dict = self.reptile_train(
                    self.model, self.optimizer, self.loss_func, 
                    self.beta, self.outer_updates, self.steps, 
                    self.info_train_loader
                    )
                params = self.export_model_parameter()
                # Update alpha to lower outer lr
                # self.alpha *= 0.993
                self.lr_scheduler.step()
                self.beta = max(self.beta * 0.993, 0.005)
                # Test before local optimization
                bf_test_dic = self.test(self.model, self.test_loader, self.loss_func, self.device)
                optim_dic = self.local_optimization(self.model, self.train_loader, self.optim_steps)
                test_dict = self.test(self.model, self.test_loader, self.loss_func, self.device)
                self.log(f"bf_test_dic: {bf_test_dic}")
                self.log(f"optim_dic: {optim_dic}")
                self.log(f"test_dict: {test_dict}")
                # send
                send_dic = {
                    # before Reptile train info
                    'bf_test_acc': test_server['test_acc'],
                    'bf_test_loss': test_server['test_loss'],
                    'bf_optim_loss': optim_server['optim_loss'],
                    'bf_optim_samples': optim_server['optim_samples'],
                    'bf_test_optim_acc': test_server_optim['test_acc'],
                    'bf_test_optim_loss': test_server_optim['test_loss'],
                    # Reptile train info
                    'train_loss': info_dict['train_loss'],
                    'train_samples': info_dict['train_samples'],
                    'train_time': info_dict['train_time'],
                    'v_info': info_dict['v_info'],
                    # after Reptile train info
                    'af_test_acc': bf_test_dic['test_acc'],
                    'af_test_loss': bf_test_dic['test_loss'],
                    'optim_loss': optim_dic['optim_loss'],
                    'optim_samples': optim_dic['optim_samples'],
                    'test_acc': test_dict['test_acc'],
                    'test_loss': test_dict['test_loss'],
                    'test_samples': test_dict['test_samples'],
                    # parameters
                    'params': params,
                    # system info
                    'num_batches': self.outer_updates * self.steps, 
                    'single_step_time': info_dict['train_time'] / (self.outer_updates * self.steps)
                }
                if self.USE_SIM_SYSHET:
                    # send_dic.update({'send_time':self.rand_time(self.communication, self.dynamics)*10})
                    send_dic.update({'send_time': self.rand_send() * 1.5})
                self.send(send_dic)
            elif data['status'] == 'EVAL':
                # The client is evaled without training.
                self.set_model_parameter(data['params'])
                test_server = self.test(self.model, self.test_loader)
                optim_server = self.local_optimization(self.model, self.train_loader, self.eval_optim_steps)
                test_server_optim = self.test(self.model, self.test_loader)
                self.log(f"test_server: {test_server}")
                self.log(f"optim_server: {optim_server}")
                self.log(f"test_server_optim: {test_server_optim}")
                self.send(data={
                    'bf_test_acc': test_server['test_acc'],
                    'bf_test_loss': test_server['test_loss'],
                    'bf_optim_loss': optim_server['optim_loss'],
                    'bf_optim_samples': optim_server['optim_samples'],
                    'bf_test_optim_acc': test_server_optim['test_acc'],
                    'bf_test_optim_loss': test_server_optim['test_loss']
                })
            elif data['status'] == 'STOP':
                self.log('stop training...')
                break
            # finish the round 
            self.finalize_round()
        # out of the loop
        self.log('stopped')


if __name__ == '__main__':
    client = MetaClient()
    client.run()