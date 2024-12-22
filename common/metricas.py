import torch
import torch.nn.functional as F

from torchmetrics.classification import AUROC
from torch.nn import BCEWithLogitsLoss

import torchmetrics 

def calculate_auc_multilabel(preds,targets,clases):
    f_auroc=torchmetrics.AUROC(task='multilabel',num_labels=len(clases),average='none')

    res=f_auroc(preds,targets.int())
    res=res.tolist()

    aucs={}
    #print('res',res)
    for c,auc in zip(clases,res):
        #print(type(c), c)
        #print('c',c)
        aucs[c]=auc
        #print(f'AUC({c}) : {auc:.3f}')
    
    return aucs

def m_accuracy_max(logits,labels):
    target=torch.argmax(labels,dim=1).long() 

    
    tam=logits.shape
    
    logits=torch.reshape(logits,(tam[0],tam[1],tam[2]*tam[3]))
    probs=F.softmax(logits,dim=1)

    count=0
    good=0
    #print('****accuracy shape:',probs.shape)
    for b in range(tam[0]):
        t =target[b]
        count +=1
        if(t>0):
            maximos=torch.max(probs[b,t,:])
            if maximos > 0.5:
                good +=1
        else:
           #print('****** probs[b,1:,:].shape:',probs[b,1:,:].shape)
           maximos=torch.max(probs[b,1:,:]) # El maximo de los defectos es 0.5
           if maximos < 0.5:
                good +=1
    return good/count

def nanF1Score(pred,target,threshold):
    y=(pred>threshold)
    t=target>threshold
    
    mask=~target.isnan()
    ymask=y[mask]
    tmask=t[mask]
    TP=torch.logical_and(ymask,tmask)
    TN=torch.logical_and(~y,~t)
    FP=torch.logical_and(ymask,~tmask)
    FN=torch.logical_and(~ymask,tmask)
    
    
    TPcol=TP.sum()
    TNcol=TN.sum()
    FPcol=FP.sum()
    FNcol=FN.sum()
    
    Prec = TPcol/(TPcol+FPcol)
    Rec= TPcol/(TPcol+FNcol)
    
    #print(Prec,Rec)
    
    f1=Prec*Rec*2/(Prec+Rec)
    return f1

def myNAN_BCElogitsLoss(preds,targets,pos_weights):
    l= pos_weights*targets*F.logsigmoid(preds) + (1-targets)*F.logsigmoid(-preds)
    return -l.mean()


def mono_accuracy(logits,labels):
    target=torch.argmax(labels,dim=1)
    output=torch.argmax(logits,dim=1)
    return torch.mean((target==output).float())

def m_ConfusionMatrix(logits,labels,cm):
    target=torch.argmax(labels,dim=1)
    output=torch.argmax(logits,dim=1)
    cm.forward(output,target)

def aucBuenoMalo(preds,targets,columna):
    
    auc=AUROC(task='binary')
    predg=preds[:,columna]
    target=targets[:,columna]
    nonans=~torch.isnan(target)
    
    return auc(predg[nonans],target[nonans])

def AUCLoss(logits,targets,pos_weights,alpha=1,c_sigmoid=1.0,topk=1):

    numclasses=targets.shape[1]
    class_losses=[]
    losses1=[]
    losses2=[]
    for clas in range(numclasses):
        t=targets[:,clas]
        o=logits[:,clas]
        nonan=~t.isnan()
        t=t[nonan]
        o=o[nonan]
        if len(t) ==0: #En este batch todos los de esta clase son nans. Pasa en validacion
            continue
        pos=(t>0.5)
        neg=(t <=.5)

        bceloss=myNAN_BCElogitsLoss(o,t,pos_weights[clas])
        losses1.append(bceloss)
        if bceloss.isnan():
            print('Nan BCE Loss in AUCLoss')
            print('output:',o)
            print('target',t)
        opos=o[pos]
        oneg=o[neg]


        aucloss=0
        if len(opos)> 0 and len(oneg)>=topk:
            oneg=torch.topk(oneg,topk)[0]
            opos=opos.view((opos.numel(),1))
            oneg=oneg.view((1,oneg.numel()))            
            diferencias=opos-oneg
            # diferencias=[]
            # for p in opos:#view
            #     for n in oneg:
            #         diferencias.append(p-n)
            # diferencias=torch.stack(diferencias)
            #aucloss=alpha*(F.relu(-diferencias)**2).mean()
            
            aucloss=alpha*myNAN_BCElogitsLoss(diferencias*c_sigmoid,torch.ones_like(diferencias),1.0)
            losses2.append(aucloss)
        class_losses.append(bceloss+aucloss)    


    losses1=torch.stack(losses1)
    
    class_losses=torch.stack(class_losses)
    loss1=losses1.mean()
    if len(losses2)>0:
        losses2=torch.stack(losses2)
        loss2=losses2.mean()
    else:
        loss2=0
    if torch.isnan(loss1):
        loss1=0
    if torch.isnan(loss2):
        loss2=0        
    final_loss=class_losses.mean()
 
    return final_loss, loss1,loss2

def AUCLoss2(logits,targets,pos_weights,alpha=0,topk=1):
    ''' Calcula una media ponderada entre el BCE medio de cada clase y el BCE de los topk ejemplos peor clasificados (con más loss)
    El loss es la media ponderada por alpha entre la media de los losses y los topk
    '''

    numclasses=targets.shape[1]
    class_losses=[]
    losses1=[]
    losses2=[]
    for clas in range(numclasses):
        t=targets[:,clas]
        o=logits[:,clas]
        nonan=~t.isnan()
        t=t[nonan]
        o=o[nonan]
        if len(t) ==0: #En este batch todos los de esta clase son nans. Pasa en validacion
            continue
        pos=(t>0.5)
        neg=(t <=.5)

        bceloss=myNAN_BCElogitsLoss(o,t,pos_weights[clas])
        losses1.append(bceloss)
        if bceloss.isnan():
            print('Nan BCE Loss in AUCLoss')
            print('output:',o)
            print('target',t)

        bce_Calculator=BCEWithLogitsLoss(reduction='none')
        if topk > len(o):# Prevenir que hayan muy pocos
            topk=len(o)
        aucloss=torch.topk( bce_Calculator(o,t), topk, largest=True)[0] # 0 valores mayores 1: indices de los valores más grandes

        aucloss=aucloss.mean()
  
        losses2.append(aucloss)
        class_losses.append( bceloss*(1-alpha)+ aucloss*alpha)    


    losses1=torch.stack(losses1)
    loss1=losses1.mean()*(1-alpha)
    losses2=torch.stack(losses2)
    loss2=losses2.mean()*alpha
        
    class_losses=torch.stack(class_losses)
    final_loss=class_losses.mean()
    if torch.isnan(loss1):
        loss1=0
    if torch.isnan(loss2):
        loss2=0        

 
    return final_loss, loss1,loss2

def mintopk(probs,k):
    tam = probs.shape
    logits_patches_reshaped = torch.reshape(probs , (tam[0],tam[1], tam[2]*tam[3] ) )
        
    top_logits_ordered = torch.topk(logits_patches_reshaped,k, dim=-1)[0]
    min_logit=torch.min(top_logits_ordered,dim=2)
    print(tam)
    print('logtis_patches_reshaped.shape',logits_patches_reshaped.shape)
    print('top_logits_ordered.shape',top_logits_ordered.shape)
    print('min_logit',min_logit[0].shape)
    return min_logit[0]

def iou(labels,probs,threshold=0.5):
    
    predbin=probs>threshold
    targetbin=labels>threshold
    intersection=(predbin*targetbin).sum()
    union=(predbin+targetbin).sum()
    if union==0:
        return 1.0
    else:
        return float((intersection/union).item())