import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from onepixel_attack import *


#just in case you decide to use Gram Schmidt to orthogonalize
def normalize(v):
    return v / np.sqrt(v.dot(v))

def schmidtty(A):
  n = len(A)

  #A[:, 0] = normalize(A[:, 0])

  for i in range(1, n):
      Ai = A[:, i]
      for j in range(0, i):
          Aj = A[:, j]
          t = Ai.dot(Aj)
          Ai = Ai - t * Aj
      A[:, i] = Ai #normalize(Ai)
    
#Finds the orthonormal vector of a given input
def ort_torch(x):
    vert=torch.randn(x.shape).to(device)
    vert-=torch.mul(vert,x)*x
    vert/=torch.norm(vert)
    return vert

def wholesale_embedding_creator(use_gpu, num_samples, reg_dset_loader_key, noisy_dset_loader_key, iters):
    if use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    base_network.to(device)
    image.to(device)
    noisy_image.to(device)

    _images, _labels, network_classified = [], [], []
    perturbed_images, perturbed_labels = [], []
    _noisy_images, noisy_ground_truth, _noisy_classified = [], [], []

    ctr = 0

    reg_test = iter(dset_loaders[reg_dset_loader_key])
    noisy_test = iter(dset_loaders[noisy_dset_loader_key])

    for i in range(0, len(dset_loaders[reg_dset_loader_key])):

        images, labels = reg_test.next()
        noisy_images, noisy_labels = noisy_test.next()

        ground_truth = torch.argmax(labels)
                
        _, logits = base_network(images)
        softmax_output = nn.Softmax(dim=1)(1.0*logits)
        predicted_class = torch.argmax(softmax_output)
        print("Ground truth: ", ground_truth)
        print("Predicted class: ", predicted_class)

        if ground_truth.item() != predicted_class.item():
            continue
        
        _, noisy_logits = base_network(noisy_images)
        noisy_softmax_output = nn.Softmax(dim=1)(1.0*noisy_logits)
        noisy_predicted_class = torch.argmax(noisy_softmax_output)
        print("Noisy predicted class: ", noisy_predicted_class)  

        draw = random.uniform(0, 1)

        if draw > .5:
          change_to = min(ground_truth.item()+1, 2)
        else:
          change_to = max(ground_truth.item()-1, 0)

        print("Change to : ", change_to)

        if change_to == ground_truth.item():
          if draw > .5:
            change_to = max(ground_truth.item()-1, 0)
          else:
            change_to = min(ground_truth.item()+1, 2)

        worked, _, _, perturbed_img = attack(base_network, images.squeeze(0).to('cuda'), true_label = ground_truth, target_label = change_to, iters = iters, verbose = True)

        if worked:
          _images.append(images)
          _labels.append(torch.argmax(labels).item())
          network_classified.append(predicted_class.item())
        
          perturbed_images.append(perturbed_img.unsqueeze(0))
          perturbed_labels.append(change_to)
        
          _noisy_images.append(noisy_images)
          noisy_ground_truth.append(torch.argmax(noisy_labels, axis = 1))
          _noisy_classified.append(noisy_predicted_class.item()) 
          
          ctr += 1 
        else:
          print(ctr)
          print("Couldn't flip :< Trying another example")

        if ctr >= num_of_samples:
          break

def church_window(use_gpu, base_network, embedding_network, image, noisy_image, groundt_label, iters):
        
    if use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    base_network.to(device)
    image.to(device)
    noisy_image.to(device)
    
    _, logits = base_network(image)  
    softmax_output = nn.Softmax(dim=1)(1.0*logits)
    predicted_class = torch.argmax(softmax_output)
    print("Ground truth: ", groundt_label)
    print("Predicted class: ", predicted_class)

    _, noisy_logits = base_network(noisy_image)
    noisy_softmax_output = nn.Softmax(dim=1)(1.0*noisy_logits)
    noisy_predicted_class = torch.argmax(noisy_softmax_output)
    print("Noisy predicted class: ", noisy_predicted_class)  

    all_lab = [0, 1, 2]
    all_lab.remove(groundt_label)
    
    regular_embedding_list = []
    noisy_embedding_list = []
    adv_embedding_list = []

    for change_to in all_lab:
        print("Flip to: ", change_to)

        worked, _, _, perturbed_img, iterations = attack(base_network, image.squeeze(0), true_label = groundt_label, target_label = change_to, iters = iters, verbose = True)

        if worked:
            print("Worked at iteration : ", iterations)

            embedding1, logits = base_network(image)
            noisy_embedding, noisy_logits = base_network(noisy_image)
            perturbed_embedding, perturbed_logits = base_network(perturbed_img.unsqueeze(0).to(device))

            cw_plotter(False, embedding_network, groundt_label, change_to, embedding1, noisy_embedding, perturbed_embedding)
            
            regular_embedding_list.append(embedding1)
            noisy_embedding_list.append(noisy_embedding)
            adv_embedding_list.append(perturbed_embedding)
            
        else:
            print("Didn't flip within chosen max iterations.")
    
    return regular_embedding_list, noisy_embedding_list, adv_embedding_list

def cw_plotter(use_gpu, embedding_network, groundt_label, change_to, embedding1, noisy_embedding, perturbed_embedding):
    LABELS = ('spiral', 'elliptical', 'merger')

    if use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    embedding_network.to(device)

    size = 1.1

    x0 = -size
    x1 = size
    y0 = -size
    y1 = size

    x = np.arange(x0, x1+.01, .01)
    y = np.arange(y0, y1+.01, .01)

    out=np.zeros([len(x),len(x),3],dtype=np.uint8)

    adv_direction = torch.subtract(perturbed_embedding.detach(), embedding1.detach())
    noisy_direction = torch.subtract(noisy_embedding.detach(), embedding1.detach())

    #you'd use Gram Schmidt here if you wanted to
    # ort = np.asarray([np.asarray(adv_direction), np.asarray(noisy_direction)])
    # schmidtty(ort)
    # np.nan_to_num(ort, nan = 0, copy = False) #unclear how else to handle division by 0

    adv_direction = adv_direction.squeeze(0) #torch.Tensor(ort[0])
    noisy_direction = noisy_direction.squeeze(0) #torch.Tensor(ort[1])

    for a in range(y.size):
        for b in range(x.size):
            update=y[a]*noisy_direction+x[b]*adv_direction
            update=update.to(device)

            image_adv=embedding1+update
            logits = embedding_network(image_adv.unsqueeze(0).to(device))
            softmax_output = nn.Softmax(dim=1)(1.0*logits)
            adv_out = torch.argmax(softmax_output)
            
            if adv_out.item() == 0:
                out[a][b]= [255, 166, 0] ##spiral: golden
            elif adv_out.item() == 1:    
                out[a][b]= [188, 80, 144] ##elliptical: dark pink
            elif adv_out.item() == 2:
                out[a][b]= [0, 63, 92] ##merger: navy
            else:
                print("something is wrong")
        
        
    my_ax = np.arange(x0, x1+.01, .01)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111)

    plt.axhline(y=len(my_ax)//2, color='k')
    plt.axvline(x=len(my_ax)//2, color='k')
    plt.imshow(out, origin = 'lower')
    plt.xticks(np.arange(len(x)), [str(round(e, 2)) for e in my_ax])
    plt.yticks(np.arange(len(y)), [str(round(e, 2)) for e in my_ax])

    for index, label in enumerate(ax.yaxis.get_ticklabels()):
        if index % 20 != 0:
          label.set_visible(False)
        
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 20 != 0:
          label.set_visible(False)

    plt.xlabel('Adversarial Direction')
    plt.ylabel('Noisy Direction')
    plt.scatter([len(my_ax)//2], [len(my_ax)//2], color = 'k', edgecolors = 'k', s = 70)
    plt.scatter([np.where((my_ax < 1.0100000e+00) & (my_ax > 1.0000000e+00))], [len(my_ax)//2], color = 'w', edgecolors = 'k', s = 70)
    plt.scatter([len(my_ax)//2], [np.where((my_ax < 1.0100000e+00) & (my_ax > 1.0000000e+00))], color = 'c', edgecolors = 'k', s = 70)

    plt.title(r'{} $\rightarrow$ {}'.format(LABELS[int(groundt_label)], LABELS[int(change_to)]))
