from __future__ import print_function
import torch
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from helper.model import Net
from helper.test import test # don't need to use full performance visuals

# note
# pillow stops working after version 6.2.2

def main():
    # Input
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    use_cuda=True

    # Define what device we are using
    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Initialize the network
    input_size = 320
    num_classes = 10
    network = Net(input_size, hidden_size=50, out_size=num_classes).to(device)

    #load trained model
    model_path = os.getcwd() + "/results/lenet_mnist_model.pth" #"/results/model.pth"
    network_state_dict = torch.load(model_path, map_location='cpu')
    network.load_state_dict(network_state_dict)

    # print model's state dict
    print("Model's state_dict:")
    for param_tensor in network.state_dict():
        print(param_tensor, "\t", network.state_dict()[param_tensor].size())

    # MNIST Test dataset and dataloader declaration
    transform = transforms.Compose([transforms.ToTensor()]) # without normalizing caused the drop to 40% accuracy
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=True)

    # MNIST Test dataset and dataloader declaration
    # transform = transforms.Compose([transforms.ToTensor(),
    #  transforms.Normalize((0.1307,), (0.3081,))]) # without normalizing caused the drop to 40% accuracy
    # testset = torchvision.datasets.MNIST(root='./data', train=False,
    #                                         download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=True)

    # testloader = torch.utils.data.DataLoader(
    # datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         ])),batch_size=1, shuffle=True)
    # best practice is to always normalize
    # decision tree and random forest does not need normalizing

    # test the performance
    testlosses = []
    test(network, testlosses, testloader)
    
    # Set the model in evaluation mode. In this case this is for the Dropout layers
    network.eval()

    # Run attack
    accuracies = []
    examples = []
    for eps in epsilons:
        acc, ex = runAttack(network, device, testloader, eps)
        accuracies.append(acc)
        examples.append(ex)

    # Accuracy vs Epsilon Results
    epsilon2AccuraciesPlot(epsilons, accuracies)

    # Plot several examples of adversarial samples at each epsilon
    samplingOverEpsilons(epsilons, examples)

def samplingOverEpsilons(epsilons, examples):
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()

def epsilon2AccuraciesPlot(epsilons, accuracies):
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

# FGSM attack code
'''
Minimize the loss by adjusting the weights based on the backpropagated gradients, 
the attack adjusts the input data to maximize the loss based on the same backpropagated gradients. 
In other words, the attack uses the gradient of the loss w.r.t the input data, 
then adjusts the input data to maximize the loss.

image - bunch of pixels
epsilon - pixel change
data_grad - minimize loss
'''
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1) # PGD needs to clamp to the rradius. A good radius is one that is not suspicious.
    # Return the perturbed image
    return perturbed_image

'''
for 1:10
    #step * step size >= radius
    clamp radius
    fgsm gradient - careful about tracking gradient for reinitializing bug
    return purturbed image

radius = 0.1 check if it is suspicious

conduct a bunch of small fgsm attacks
'''
def pgd_attack(image, epsilon, data_grad):
    alpha = epsilon/10
    perturbed_image = image

    for x in range(0,10):
        perturbed_image = fgsm_attack(perturbed_image, alpha, data_grad) # the data grad should be diff on each step

    return perturbed_image

def runAttack( model,device,test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    correct = loopOverTestSet(test_loader, device, model, epsilon, adv_examples, correct)

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples

# taking a step with a calc gradient
# 
def loopOverTestSet(test_loader, device, model, epsilon, adv_examples, correct):
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        ## this and down goes into pgd attack
        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)
        #perturbed_data = pgd_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct = checkForSuccess(final_pred, target, epsilon, adv_examples, perturbed_data, init_pred, correct)
    return correct

def checkForSuccess(final_pred, target, epsilon, adv_examples, perturbed_data, init_pred, correct):
    if final_pred.item() == target.item():
        correct += 1
        # Special case for saving 0 epsilon examples
        if (epsilon == 0) and (len(adv_examples) < 5):
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
    else:
        # Save some adv examples for visualization later
        if len(adv_examples) < 5:
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
    return correct


if __name__ == '__main__':
    main()
