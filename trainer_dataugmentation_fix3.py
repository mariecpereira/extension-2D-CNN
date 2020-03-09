# -*- coding: utf-8 -*-
import os
import copy
import time
import pickle
import numpy as np
import random, math
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
from IPython import display


class DeepNetTrainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, data_transf=None,
                 lr_scheduler=None, callbacks=None, use_gpu='auto'):

        assert (model is not None) and (criterion is not None) and (optimizer is not None)
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.data_transf = data_transf
        self.scheduler = lr_scheduler
        #self.metrics = dict(train=dict(losses=[]), valid=dict(losses=[]))
        self.metrics = dict(train=OrderedDict(losses=[]), valid=OrderedDict(losses=[]))
        self.last_epoch = 0

        self.callbacks = []
        if callbacks is not None:
            for cb in callbacks:
                self.callbacks.append(cb)
                cb.trainer = self

        self.use_gpu = use_gpu
        if use_gpu == 'auto':
            self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.model.cuda()

    def fit(self, n_epochs, Xin, Yin, valid_data=None, valid_split=None, batch_size=10, shuffle=True):
        if valid_data is not None:
            train_loader = DataLoader(create_dataloaders(Xin, Yin, transform=self.data_transf), 
                                      batch_size=batch_size, shuffle=shuffle)
            valid_loader = DataLoader(create_dataloaders(*valid_data, transform=self.data_transf),
                                      batch_size=batch_size, shuffle=shuffle)
        elif valid_split is not None:
            iv = int(valid_split * Xin.shape[0])
            Xval, Yval = Xin[:iv], Yin[:iv]
            Xtra, Ytra = Xin[iv:], Yin[iv:]
            train_loader = DataLoader(create_dataloaders(Xtra, Ytra, transform=data_transf), batch_size=batch_size, shuffle=shuffle)
            valid_loader = DataLoader(create_dataloaders(Xval, Yval, transform=data_transf), batch_size=batch_size, shuffle=shuffle)
        else:
            train_loader = DataLoader(create_dataloaders(Xin, Yin, transform=data_transf), batch_size=batch_size, shuffle=shuffle)
            valid_loader = None
        train_loader
        valid_loader
        self.fit_loader(n_epochs, train_loader, valid_data=valid_loader)

    def score(self, Xin, Yin, batch_size=10):
        dloader = DataLoader(TensorDataset(Xin, Yin), batch_size=batch_size, shuffle=False)
        return self.score_loader(dloader)

    def evaluate(self, Xin, Yin, metrics=None, batch_size=10):
        dloader = DataLoader(TensorDataset(Xin, Yin), batch_size=batch_size, shuffle=False)
        return self.evaluate_loader(dloader, metrics)

    def fit_loader(self, n_epochs, train_data, valid_data=None):
        self.has_validation = valid_data is not None
        self.n_batches = len(train_data.dataset)//train_data.batch_size # AQUIIII
        try:
            for cb in self.callbacks:
                cb.on_train_begin(n_epochs, self.metrics)

            # for each epoch
            for curr_epoch in range(self.last_epoch + 1, self.last_epoch + n_epochs + 1):

                # training phase
                # ==============
                for cb in self.callbacks:
                    cb.on_epoch_begin(curr_epoch, self.metrics)

                epo_samples = 0
                epo_batches = 0
                epo_loss = 0

                self.model.train(True)
                if self.scheduler is not None:
                    self.scheduler.step()

                # for each minibatch
                for curr_batch, (X, Y) in enumerate(train_data):
                    Y = Y.squeeze(-1)
                    mb_size = X[0].size(0)
                    #print('MB size:',mb_size)
                    epo_samples += mb_size
                    epo_batches += 1

                    for cb in self.callbacks:
                        cb.on_batch_begin(curr_epoch, curr_batch, mb_size)

                    if self.use_gpu:
                        X, Y = Variable(X.cuda()), Variable(Y.cuda())
                    else:
                        X, Y = Variable(X), Variable(Y)

                    self.optimizer.zero_grad()

                    Ypred = self.model.forward(X)
                    loss = self.criterion(Ypred, Y)
                    loss.backward()
                    self.optimizer.step()

                    #vloss = loss.data.cpu()[0]
                    vloss = loss.data.cpu().item()
                    if hasattr(self.criterion, 'size_average') and self.criterion.size_average:
                        epo_loss += mb_size * vloss
                        #print(epo_loss)
                    else:
                        epo_loss += vloss

                    for cb in self.callbacks:
                        cb.on_batch_end(curr_epoch, curr_batch, Ypred, Y, loss)

                # end of training minibatches
                #eloss = float(epo_loss / epo_samples)
                #print(eloss)
                self.train_loss = float(epo_loss / epo_samples)
                self.metrics['train']['losses'].append(self.train_loss)
                #print(self.metrics)

                # validation phase
                # ================
                if self.has_validation:
                    epo_samples = 0
                    epo_batches = 0
                    epo_loss = 0

                    self.model.train(False)

                    # for each minibatch
                    for curr_batch, (X, Y) in enumerate(valid_data):
                        Y = Y.squeeze(-1)
                        mb_size = X[0].size(0)
                        epo_samples += mb_size
                        epo_batches += 1

                        for cb in self.callbacks:
                            cb.on_vbatch_begin(curr_epoch, curr_batch, mb_size)

                        if self.use_gpu:
                            X, Y = Variable(X.cuda()), Variable(Y.cuda())
                        else:
                            X, Y = Variable(X), Variable(Y)

                        Ypred = self.model.forward(X)
                        loss = self.criterion(Ypred, Y)

                        #vloss = loss.data.cpu()[0]
                        vloss = loss.data.cpu().item()
                        if hasattr(self.criterion, 'size_average') and self.criterion.size_average:
                            epo_loss += vloss * mb_size
                        else:
                            epo_loss += vloss

                        for cb in self.callbacks:
                            cb.on_vbatch_end(curr_epoch, curr_batch, Ypred, Y, loss)

                    # end minibatches
                    eloss = float(epo_loss / epo_samples)
                    self.valid_loss = float(epo_loss / epo_samples)
                    self.metrics['valid']['losses'].append(self.valid_loss)
                    #print(self.metrics)

                else:
                    self.metrics['valid']['losses'].append(None)

                for cb in self.callbacks:
                    cb.on_epoch_end(curr_epoch, self.metrics)

        except KeyboardInterrupt:
            pass

        for cb in self.callbacks:
            cb.on_train_end(n_epochs, self.metrics)

    def score_loader(self, data_loader):
        epo_samples = 0
        epo_loss = 0
        self.model.train(False)
        for curr_batch, (X, Y) in enumerate(data_loader):
            mb_size = X[0].size(0)
            epo_samples += mb_size
            if self.use_gpu:
                X, Y = Variable(X.cuda()), Variable(Y.cuda())
            else:
                X, Y = Variable(X), Variable(Y)
            Ypred = self.model.forward(X)
            loss = self.criterion(Ypred, Y)
            vloss = loss.data.cpu()[0]
            #vloss = loss.data.cpu().item()
            if hasattr(self.criterion, 'size_average') and self.criterion.size_average:
                epo_loss += vloss * mb_size
            else:
                epo_loss += vloss
        epo_loss /= epo_samples
        # higher score is better
        return -epo_loss

    def evaluate_loader(self, data_loader, metrics=None, verbose=1):
        metrics = metrics or []
        my_metrics = dict(train=dict(losses=[]), valid=dict(losses=[]))
        for cb in metrics:
            cb.on_train_begin(1, my_metrics)
            cb.on_epoch_begin(1, my_metrics)

        epo_samples = 0
        epo_batches = 0
        epo_loss = 0

        try:
            self.model.train(False)
            ii_n = len(data_loader)
            

            for curr_batch, (X, Y) in enumerate(data_loader):
                mb_size = X[0].size(0)
                epo_samples += mb_size
                epo_batches += 1

                if self.use_gpu:
                    X, Y = Variable(X.cuda()), Variable(Y.cuda())
                else:
                    X, Y = Variable(X), Variable(Y)

                Ypred = self.model.forward(X)
                loss = self.criterion(Ypred, Y)

                #vloss = loss.data.cpu()[0]
                vloss = loss.data.cpu().item()
                if hasattr(self.criterion, 'size_average') and self.criterion.size_average:
                    epo_loss += vloss * mb_size
                else:
                    epo_loss += vloss

                for cb in metrics:
                    cb.on_batch_end(1, curr_batch, Ypred, Y, loss)

                #print('\revaluate: {}/{}'.format(curr_batch, ii_n - 1), end='')
                print('\revaluate: {}/{}'.format(curr_batch, ii_n - 1))
            print(' ok')

        except KeyboardInterrupt:
            print(' interrupted!')

        if epo_batches > 0:
            epo_loss /= epo_samples
            my_metrics['train']['losses'].append(epo_loss)
            for cb in metrics:
                cb.on_epoch_end(1, my_metrics)

        #return dict([(k, v[0]) for k, v in my_metrics['train'].items()])
        return my_metrics['valid']

    def load_state(self, file_basename):
        load_trainer_state(file_basename, self.model, self.metrics)

    def save_state(self, file_basename):
        save_trainer_state(file_basename, self.model, self.metrics)

    def predict(self, Xin, Yin):
        if self.use_gpu:
            Xin = Xin.cuda()
        return predict(self.model, Xin, Yin, self.data_transf)

    def predict_classes(self, Xin):
        if self.use_gpu:
            Xin = Xin.cuda()
        return predict_classes(self.model, Xin)

    def predict_probas(self, Xin):
        if self.use_gpu:
            Xin = Xin.cuda()
        return predict_probas(self.model, Xin)
    
    def summary(self):
        pass

def load_trainer_state(file_basename, model, metrics):
    model.load_state_dict(torch.load(file_basename + '.model'))
    if os.path.isfile(file_basename + '.histo'):
        metrics.update(pickle.load(open(file_basename + '.histo', 'rb')))


def save_trainer_state(file_basename, model, metrics):
    torch.save(model.state_dict(), file_basename + '.model')
    pickle.dump(metrics, open(file_basename + '.histo', 'wb'))


def predict(model, Xin, Yin, data_transf):
    valid_loader = DataLoader(create_dataloaders(Xin, Yin, transform=data_transf), batch_size=Xin.shape[0], shuffle=False)
    X, Y = next(iter(valid_loader))
    y_pred = model.forward(Variable(X))
    return y_pred.data


def predict_classes(model, Xin):
    y_pred = predict(model, Xin)
    _, pred = torch.max(y_pred, 1)
    return pred

def predict_probas(model, Xin):
    y_pred = predict(model, Xin)
    probas = F.softmax(y_pred)
    return probas

class create_dataloaders(torch.utils.data.Dataset):
    def __init__(self, x, y=None, transform=None):
        super().__init__()
        self.transform = transform
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.size()[0]#self.x.shape[0]
    def __getitem__(self, idx):
        if self.transform is not None:
            xi = self.transform(self.x[idx])[0]#.repeat(3, 1, 1)#.repeat(1, 3, 1).permute(1, 0, 2)
        else:
            xi = self.x[idx]
        if self.y is None:
            yi = None
        else:
            yi = torch.LongTensor(self.y[idx])
            #yi = torch.Tensor(self.y[idx])
        return xi, yi
    
class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        #image = image.permute(1, 0, 2)
        h, w = image.size()[-2:]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        n_image = image[:, top: top + new_h, left: left + new_w]
        topt = top + new_h
        leftt = left + new_w
        return n_image , top, topt, left, leftt 
    
class GetCord(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        #image = image.permute(1, 0, 2)
        h, w = image.size()[-2:]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        n_image = image[:, top: top + new_h, left: left + new_w]
        topt = top + new_h
        leftt = left + new_w
        return top 
    
class Crop1(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        #image = image.permute(1, 0, 2)
        h, w = image.size()[-2:]
        new_h, new_w = self.output_size
        #top = np.random.randint(0, h - new_h)
        #left = np.random.randint(0, w - new_w)
        #new_image = image[:,-80:,15:95]
        new_image = image[:,30:110,30:110]
        #image = image[:, np.ceil(h/2):top + new_h, left: left + new_w]
        image = image/(torch.max(image))
        return new_image    
    
class Crop2(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        #image = image.permute(1, 0, 2)
        h, w = image.size()[-2:]
        hh = 70
        ww = 70
        new_h, new_w = self.output_size
        new_image2 = image[:, hh-(new_h/2):(hh-(new_h/2))+new_h, ww-(new_w/2):(ww-(new_w/2))+new_w]
        #image = image/(torch.max(image)+0.00000000000000000001)
        return new_image2     
    
class Flip(object):
    def __init__(self):
        pass

    def __call__(self, image):
        rand_op = random.choice([0,1])
        
        if rand_op == 0:
            #print(image.shape)
            dim = 2
            dim = image.dim() + dim if dim < 0 else dim
            inds = tuple(slice(None, None) if i != dim 
                         else image.new(torch.arange(image.size(i)-1, -1, -1).tolist()).long()
                         for i in range(image.dim()))
            image[inds]
        else:
            image = image

        return image

def th_iterproduct(*args):
    return torch.from_numpy(np.indices(args).reshape((len(args),-1)).T)

def th_nearest_interp2d(input, coords):
    """
    2d nearest neighbor interpolation th.Tensor
    """
    # take clamp of coords so they're in the image bounds
    x = torch.clamp(coords[:,:,0], 0, input.size(1)-1).round()
    y = torch.clamp(coords[:,:,1], 0, input.size(2)-1).round()

    stride = torch.cuda.LongTensor(input.stride())
    x_ix = x.mul(stride[1]).long()
    y_ix = y.mul(stride[2]).long()

    input_flat = input.view(input.size(0),-1)

    mapped_vals = input_flat.gather(1, x_ix.add(y_ix))

    return mapped_vals.view_as(input)

def th_bilinear_interp2d(input, coords):
    """
    bilinear interpolation in 2d
    """
    x = torch.clamp(coords[:,:,0], 0, input.size(1)-2)
    x0 = x.floor()
    x1 = x0 + 1
    y = torch.clamp(coords[:,:,1], 0, input.size(2)-2)
    y0 = y.floor()
    y1 = y0 + 1

    stride = torch.cuda.LongTensor(input.stride())
    x0_ix = x0.mul(stride[1]).long()
    x1_ix = x1.mul(stride[1]).long()
    y0_ix = y0.mul(stride[2]).long()
    y1_ix = y1.mul(stride[2]).long()

    x0_ix = x0_ix.cuda()
    x1_ix = x1_ix.cuda()
    y0_ix = y0_ix.cuda()
    y1_ix = y1_ix.cuda()
    
    input_flat = input.view(input.size(0),-1)

    vals_00 = input_flat.gather(1, x0_ix.add(y0_ix))
    vals_10 = input_flat.gather(1, x1_ix.add(y0_ix))
    vals_01 = input_flat.gather(1, x0_ix.add(y1_ix))
    vals_11 = input_flat.gather(1, x1_ix.add(y1_ix))
    
    xd = (x - x0).cuda()
    yd = (y - y0).cuda()
    xm = (1 - xd).cuda()
    ym = (1 - yd).cuda()

    x_mapped = (vals_00.mul(xm).mul(ym) +
                vals_10.mul(xd).mul(ym) +
                vals_01.mul(xm).mul(yd) +
                vals_11.mul(xd).mul(yd))

    return x_mapped.view_as(input)


def th_affine2d(x, matrix, mode='bilinear', center=True):
    """
    2D Affine image transform on th.Tensor
    
    Arguments
    ---------
    x : th.Tensor of size (C, H, W)
        image tensor to be transformed
    matrix : th.Tensor of size (3, 3) or (2, 3)
        transformation matrix
    mode : string in {'nearest', 'bilinear'}
        interpolation scheme to use
    center : boolean
        whether to alter the bias of the transform 
        so the transform is applied about the center
        of the image rather than the origin
    Example
    ------- 
    >>> import torch
    >>> from torchsample.utils import *
    >>> x = th.zeros(2,1000,1000)
    >>> x[:,100:1500,100:500] = 10
    >>> matrix = th.FloatTensor([[1.,0,-50],
    ...                             [0,1.,-50]])
    >>> xn = th_affine2d(x, matrix, mode='nearest')
    >>> xb = th_affine2d(x, matrix, mode='bilinear')
    """

    if matrix.dim() == 2:
        matrix = matrix[:2,:]
        matrix = matrix.unsqueeze(0)
    elif matrix.dim() == 3:
        if matrix.size()[1:] == (3,3):
            matrix = matrix[:,:2,:]

    A_batch = matrix[:,:,:2]
    if A_batch.size(0) != x.size(0):
        A_batch = A_batch.repeat(x.size(0),1,1)
    b_batch = matrix[:,:,2].unsqueeze(1)

    # make a meshgrid of normal coordinates
    _coords = th_iterproduct(x.size(1),x.size(2))
    coords = _coords.unsqueeze(0).repeat(x.size(0),1,1).float()

    if center:
        # shift the coordinates so center is the origin
        coords[:,:,0] = coords[:,:,0] - (x.size(1) / 2. - 0.5)
        coords[:,:,1] = coords[:,:,1] - (x.size(2) / 2. - 0.5)
    # apply the coordinate transformation
    new_coords = coords.bmm(A_batch.transpose(1,2)) + b_batch.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,:,0] = new_coords[:,:,0] + (x.size(1) / 2. - 0.5)
        new_coords[:,:,1] = new_coords[:,:,1] + (x.size(2) / 2. - 0.5)

    # map new coordinates using bilinear interpolation
    if mode == 'nearest':
        x_transformed = th_nearest_interp2d(x.contiguous(), new_coords)
    elif mode == 'bilinear':
        x_transformed = th_bilinear_interp2d(x.contiguous(), new_coords)

    return x_transformed

class Rotate(object):
    def __init__(self, 
                 value,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.
        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        self.value = value
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        theta = math.pi / 180 * self.value
        rotation_matrix = torch.FloatTensor([[math.cos(theta), -math.sin(theta), 0],
                                          [math.sin(theta), math.cos(theta), 0],
                                          [0, 0, 1]])
        if self.lazy:
            return rotation_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       rotation_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]
        
class RandomRotate(object):
    def __init__(self, 
                 rotation_range,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.
        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees
        interp : string in {'bilinear', 'nearest'} or list of strings
            type of interpolation to use. You can provide a different
            type of interpolation for each input, e.g. if you have two
            inputs then you can say `interp=['bilinear','nearest']
        lazy    : boolean
            if true, only create the affine transform matrix and return that
            if false, perform the transform on the tensor and return the tensor
        """
        self.rotation_range = rotation_range
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        degree = random.uniform(-self.rotation_range, self.rotation_range)

        if self.lazy:
            return Rotate(degree, lazy=True)(inputs[0])
        else:
            outputs = Rotate(degree,
                             interp=self.interp)(*inputs)
        return outputs
    
class Callback(object):
    def __init__(self):
        pass

    def on_train_begin(self, n_epochs, metrics):
        pass

    def on_train_end(self, n_epochs, metrics):
        pass

    def on_epoch_begin(self, epoch, metrics):
        pass

    def on_epoch_end(self, epoch, metrics):
        pass

    def on_batch_begin(self, epoch, batch, mb_size):
        pass

    def on_batch_end(self, epoch, batch, y_pred, y_true, loss):
        pass

    def on_vbatch_begin(self, epoch, batch, mb_size):
        pass

    def on_vbatch_end(self, epoch, batch, y_pred, y_true, loss):
        pass


class AccuracyMetric(Callback):
    def __init__(self):
        super().__init__()
        self.name = 'acc'

    def on_batch_end(self, epoch_num, batch_num, y_pred, y_true, loss):
        _, preds = torch.max(y_pred.data, 1)
        ok = (preds == y_true.data).sum()
        #self.train_accum += ok
        self.train_accum += ok.item()
        self.n_train_samples += y_pred.size(0)

    def on_vbatch_end(self, epoch_num, batch_num, y_pred, y_true, loss):
        _, preds = torch.max(y_pred.data, 1)
        ok = (preds == y_true.data).sum()
        #self.valid_accum += ok
        self.train_accum += ok.item()
        self.n_valid_samples += y_pred.size(0)

    def on_epoch_begin(self, epoch_num, metrics):
        self.train_accum = 0
        self.valid_accum = 0
        self.n_train_samples = 0
        self.n_valid_samples = 0

    def on_epoch_end(self, epoch_num, metrics):
        if self.n_train_samples > 0:
            metrics['train'][self.name].append(1.0 * self.train_accum / self.n_train_samples)
        if self.n_valid_samples > 0:
            metrics['valid'][self.name].append(1.0 * self.valid_accum / self.n_valid_samples)

    def on_train_begin(self, n_epochs, metrics):
        metrics['train'][self.name] = []
        metrics['valid'][self.name] = []


class ModelCheckpoint(Callback):

    def __init__(self, model_basename, reset=False, verbose=0):
        super().__init__()
        os.makedirs(os.path.dirname(model_basename), exist_ok=True)
        self.basename = model_basename
        self.reset = reset
        self.verbose = verbose

    def on_train_begin(self, n_epochs, metrics):
        if (self.basename is not None) and (not self.reset) and (os.path.isfile(self.basename + '.model')):
            load_trainer_state(self.basename, self.trainer.model, self.trainer.metrics)
            if self.verbose > 0:
                print('Model loaded from', self.basename + '.model')

        self.trainer.last_epoch = len(self.trainer.metrics['train']['losses'])
        if self.trainer.scheduler is not None:
            self.trainer.scheduler.last_epoch = self.trainer.last_epoch

        self.best_model = copy.deepcopy(self.trainer.model)
        self.best_epoch = self.trainer.last_epoch
        self.best_loss = 1e10
        if self.trainer.last_epoch > 0:
            self.best_loss = self.trainer.metrics['valid']['losses'][-1] or self.trainer.metrics['train']['losses'][-1]

    def on_train_end(self, n_epochs, metrics):
        if self.verbose > 0:
            print('Best model was saved at epoch {} with loss {:.5f}: {}'
                  .format(self.best_epoch, self.best_loss, self.basename))

    def on_epoch_end(self, epoch, metrics):
        eloss = metrics['valid']['losses'][-1] or metrics['train']['losses'][-1]
        if eloss < self.best_loss:
            self.best_loss = eloss
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(self.trainer.model)
            if self.basename is not None:
                save_trainer_state(self.basename, self.trainer.model, self.trainer.metrics)
                if self.verbose > 1:
                    print('Model saved to', self.basename + '.model')


class PrintCallback(Callback):

    def __init__(self):
        super().__init__()

    def on_train_begin(self, n_epochs, metrics):
        print('Start training for {} epochs'.format(n_epochs))

    def on_train_end(self, n_epochs, metrics):
        n_train = len(metrics['train']['losses'])
        print('Stop training at epoch: {}/{}'.format(n_train, self.trainer.last_epoch + n_epochs))

    def on_epoch_begin(self, epoch, metrics):
        self.t0 = time.time()

    def on_epoch_end(self, epoch, metrics):
            is_best = ''
            has_valid = len(metrics['valid']['losses']) > 0 and metrics['valid']['losses'][0] is not None
            has_metrics = len(metrics['train'].keys()) > 1
            etime = time.time() - self.t0

            if has_valid:
                if epoch == int(np.argmin(metrics['valid']['losses'])) + 1:
                    is_best = 'best'
                if has_metrics:
                    # validation and metrics
                    metric_name = [mn for mn in metrics['valid'].keys() if mn != 'losses'][0]
                    # metric_name = list(self.trainer.compute_metric.keys())[0]
                    print('{:3d}: {:5.1f}s   T: {:.5f} {:.5f}   V: {:.5f} {:.5f} {}'
                          .format(epoch, etime,
                                  metrics['train']['losses'][-1],
                                  metrics['train'][metric_name][-1],
                                  metrics['valid']['losses'][-1],
                                  metrics['valid'][metric_name][-1], is_best))
                else:
                    # validation and no metrics
                    print('{:3d}: {:5.1f}s   T: {:.5f}   V: {:.5f} {}'
                          .format(epoch, etime,
                                  metrics['train']['losses'][-1],
                                  metrics['valid']['losses'][-1], is_best))
            else:
                if epoch == int(np.argmin(metrics['train']['losses'])) + 1:
                    is_best = 'best'
                if has_metrics:
                    # no validation and metrics
                    metric_name = list(self.trainer.compute_metric.keys())[0]
                    print('{:3d}: {:5.1f}s   T: {:.5f} {:.5f} {}'
                          .format(epoch, etime,
                                  metrics['train']['losses'][-1],
                                  metrics['train'][metric_name][-1], is_best))
                else:
                    # no validation and no metrics
                    print('{:3d}: {:5.1f}s   T: {:.5f} {}'
                          .format(epoch, etime,
                                  metrics['train']['losses'][-1], is_best))


class PlotCallback(Callback):
    def __init__(self, interval=1, max_loss=None):
        super().__init__()
        self.interval = interval
        self.max_loss = max_loss

    def on_train_begin(self, n_epochs, metrics):
        self.line_train = None
        self.line_valid = None
        self.dot_train = None
        self.dot_valid = None

        self.fig = plt.figure(figsize=(15, 6))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.grid(True)

        self.plot_losses(self.trainer.metrics['train']['losses'],
                         self.trainer.metrics['valid']['losses'])

    def on_epoch_end(self, epoch, metrics):
        if epoch % self.interval == 0:
            display.clear_output(wait=True)
            self.plot_losses(self.trainer.metrics['train']['losses'],
                             self.trainer.metrics['valid']['losses'])

    def plot_losses(self, htrain, hvalid):
        epoch = len(htrain)
        if epoch == 0:
            return

        x = np.arange(1, epoch + 1)
        if self.line_train:
            self.line_train.remove()
        if self.dot_train:
            self.dot_train.remove()
        self.line_train, = self.ax.plot(x, htrain, color='#1f77b4', linewidth=2, label='training loss')
        best_epoch = int(np.argmin(htrain)) + 1
        best_loss = htrain[best_epoch - 1]
        self.dot_train = self.ax.scatter(best_epoch, best_loss, c='#1f77b4', marker='o')

        if hvalid[0] is not None:
            if self.line_valid:
                self.line_valid.remove()
            if self.dot_valid:
                self.dot_valid.remove()
            self.line_valid, = self.ax.plot(x, hvalid, color='#ff7f0e', linewidth=2, label='validation loss')
            best_epoch = int(np.argmin(hvalid)) + 1
            best_loss = hvalid[best_epoch - 1]
            self.dot_valid = self.ax.scatter(best_epoch, best_loss, c='#ff7f0e', marker='o')

        self.ax.legend()
        # self.ax.vlines(best_epoch, *self.ax.get_ylim(), colors='#EBDDE2', linestyles='dashed')
        self.ax.set_title('Best epoch: {}, Current epoch: {}'.format(best_epoch, epoch))

        display.display(self.fig)
        time.sleep(0.1)