from . import AlexNet, MobileNet, CNN, RNN, VGG, ResNet


def create_model_instance_custom(model_type, dataset_type):
    # femnist, cifar10: FedAvgCNN
    # 
    # shakespeare: RNN
    model = None
    if model_type == 'fedavgcnn':
        if dataset_type == 'cifar10':
            model = CNN.FedAvgCNN(in_features=3, num_classes=10, dim=1600)
        elif dataset_type == 'emnist' or dataset_type == 'femnist':
            model = CNN.FedAvgCNN(in_features=1, num_classes=62)
    elif model_type == 'alexnet':
        if dataset_type == 'cifar10':
            model = AlexNet.AlexNet()
    elif model_type == 'alexnet_drop':
        if dataset_type == 'cifar10':
            model = AlexNet.AlexNet_drop()
    elif model_type == 'lstm':
        model = RNN.CharLSTM()
    if model is None:
        raise Exception('Unknown model_type: ' + model_type + ' and dataset_type: ' + dataset_type)
    else:
        return model