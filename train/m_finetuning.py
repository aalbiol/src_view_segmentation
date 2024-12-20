from pytorch_lightning.callbacks import BaseFinetuning

# Freeze all layers except one
def freeze_all_layers_except_one(model, layer_name):
    for name, param in model.named_parameters():
        if layer_name not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def all_layers_except_one(model, layer_name):
    todos_menos_uno = []
    uno=[]
    for name, param in model.named_parameters():
        if layer_name not in name:
            todos_menos_uno.append(param)
        else:
            uno.append(param)
    return todos_menos_uno, uno

def count_unfrozen_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10,initial_denom_lr=2):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.initial_denom_lr=initial_denom_lr

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        self.freeze(pl_module.modelo.features)
        self.lista = list(pl_module.modelo.model.classifier.children()) 
        for i in range(len(self.lista)-1):
            self.freeze(self.lista[i])

    def finetune_function(self, pl_module, current_epoch, optimizer):
    # When `current_epoch` is 10, feature_extractor will start training.
        submodulos=list(pl_module.modelo.features.children())
        submodulos.append(self.lista[:-1])
        nsubmodulos=len(submodulos)
        for i in range(1,nsubmodulos):
                if current_epoch == i* self._unfreeze_at_epoch:
                    self.unfreeze_and_add_param_group(
                    #modules=pl_module.modelo.features.layer4,
                    modules=submodulos[-i],
                    optimizer=optimizer,
                    train_bn=True,
                    initial_denom_lr=self.initial_denom_lr)
                    unfrozen_params=count_unfrozen_parameters(pl_module.modelo)
                    print(f"Unfreezing layer {nsubmodulos -i } Learning {unfrozen_params} parameters")

class RefineLastLayerFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10,initial_denom_lr=2):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.initial_denom_lr=initial_denom_lr

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        self.freeze(pl_module.modelo.features)
        self.lista = list(pl_module.modelo.model.classifier.children()) 
        for i in range(len(self.lista)-1):
            self.freeze(self.lista[i])


class FeatureExtractorFreezeUnfreeze2(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10,initial_denom_lr=2):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.initial_denom_lr=initial_denom_lr

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        self.freeze(pl_module.modelo.features)

    def finetune_function(self, pl_module, current_epoch, optimizer):
    # When `current_epoch` is 10, feature_extractor will start training.
        submodulos=list(pl_module.modelo.features.children())
        nsubmodulos=len(submodulos)
        for i in range(1,nsubmodulos):
                if current_epoch == i* self._unfreeze_at_epoch:
                    self.unfreeze_and_add_param_group(
                    #modules=pl_module.modelo.features.layer4,
                    modules=submodulos[-i],
                    optimizer=optimizer,
                    train_bn=True,
                    initial_denom_lr=self.initial_denom_lr)
                    unfrozen_params=count_unfrozen_parameters(pl_module.modelo)
                    print(f"Unfreezing layer {nsubmodulos -i } Learning {unfrozen_params} parameters")