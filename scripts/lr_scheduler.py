import math
import matplotlib.pyplot as plt


def lr_time_based_decay(epoch, lr, nb_epoch=0):
    # decay = lr / nb_epoch
    decay = 0.0045
    lr *= 1 / (1 + decay * epoch)
    return lr


def lr_step_decay(epoch, lr, nb_epoch=0):
    drop_rate = .99
    epochs_drop = 8

    if epoch % epochs_drop == 0:
        return lr * math.pow(drop_rate, math.floor(epoch/epochs_drop))

    return lr


def lr_asc_desc_decay(epoch, lr, nb_epoch=0):
    lr_max = 1e-3
    lr_min = 1e-5
    lr_ascending_ep = 100
    lr_sus_ep = 0 # plato
    decay = 0.99
    ascending_penalty = 0.95

    if epoch < lr_ascending_ep:
        lr = (lr_max - lr) / lr_ascending_ep * epoch + (lr*ascending_penalty)

    elif epoch < lr_ascending_ep + lr_sus_ep:
        lr = lr_max

    else:
        lr = (lr_max - lr_min) * decay**(
            epoch - lr_ascending_ep - lr_sus_ep
        ) + lr_min

    return lr


def lr_triangular_decay(epoch, lr, nb_epoch=0):
    ...


def lr_epoch_fall(epoch, lr, nb_epoch=0):
    lr_drop = 20
    lr = lr * (0.5 ** (epoch // lr_drop))
    return lr


def lr_exp_decay(epoch, lr, nb_epoch=0):
    lr = lr * math.exp(-0.001 * epoch)
    return lr


def plot_lr_decay(lr_function, lr, epoch, nb_epoch):
    if epoch == nb_epoch:
        return lr
    else:
        lr.append(lr_function(epoch, lr[-1], nb_epoch))
        return plot_lr_decay(lr_function, lr, epoch+1, nb_epoch)


if __name__ == "__main__":
    nb_epoch = 500
    returned_lr = plot_lr_decay(
        lr_asc_desc_decay,
        [1e-5],
        0,
        nb_epoch
    )
    print(returned_lr)

    plt.plot(list(range(0, (nb_epoch+1))), returned_lr)
    plt.savefig('lr_decay.png')
    plt.show()
