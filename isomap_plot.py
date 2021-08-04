#https://pythonmatplotlibtips.blogspot.com/2018/01/rotate-azimuth-angle-animation-3d-python-matplotlib-pyplot.html
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.manifold import Isomap
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



def isomap_plot(base_network, img_reg, img_noise, img_pert, num, three_dim=True, save=False, where=False):
  
  ind = np.random.randint(0,len(img_reg),num)
  num2 = 2 * num
    
  base_network.to('cpu')
  embedding_reg, logits_reg = base_network(img_reg[ind])
  embedding_noise, logits_noise = base_network(img_noise[ind])
  embedding_pert, logits_pert = base_network(img_pert)

  total = []
  total.append(embedding_reg)
  total.append(embedding_noise)
  total.append(embedding_pert)

  total = torch.cat(total)
  len(total)

  softmax_reg = nn.Softmax(dim=1)(1.0 * logits_reg)
  _, predict_reg = torch.max(softmax_reg, 1)

  softmax_noi = nn.Softmax(dim=1)(1.0 * logits_noise)
  _, predict_noi = torch.max(softmax_noi, 1)

  softmax_pert = nn.Softmax(dim=1)(1.0 * logits_pert)
  _, predict_pert = torch.max(softmax_pert, 1)


  predict_reg = predict_reg.tolist()
  predict_noi = predict_noi.tolist()
  predict_pert = predict_pert.tolist()

  color = np.array(['r', 'b', 'k'])

  if three_dim:
    model = Isomap(n_components=3)

    fig = plt.figure(figsize = (20, 10))
    ax = plt.gca()
    ax = plt.axes(projection='3d')
    plt.title('Isomap 3D \n Curves: Y10 (orange), Y1 noisy (blue) \n Point color - predicted class: spiral (red), elliptical (blue), merger (black) \n Perturbed images - large points', fontsize=25)

    proj = model.fit_transform(total.detach().numpy())
    ax.plot_trisurf(proj[:num, 0], proj[:num, 1], proj[:num, 2], cmap="autumn", alpha = 0.3)
    ax.plot_trisurf(proj[num:num2, 0], proj[num:num2, 1], proj[num:num2, 2],cmap="winter", alpha=0.3)
    ax.plot_trisurf(proj[num2:, 0], proj[num2:, 1], proj[num2:, 2],cmap="Reds", alpha=0.3)

    ax.scatter(proj[:num, 0], proj[:num, 1], proj[:num, 2], c=color[predict_reg[:]],marker='o', s=40)
    ax.scatter(proj[num:num2, 0], proj[num:num2, 1], proj[num:num2, 2],c=color[predict_noi[:]],marker='o', s=40)
    ax.scatter(proj[num2:, 0], proj[num2:, 1], proj[num2:, 2],c=color[predict_pert[:]],marker='^', s=250)

  else:
    model = Isomap(n_components=2)

    fig = plt.figure(figsize = (20, 10))
    ax = plt.gca()
    plt.title('Isomap 2D \n Point color - predicted class: spiral (red), elliptical (blue), merger (black) \n Perturbed images - large points', fontsize=25)

    proj = model.fit_transform(total.detach().numpy())
    ax.scatter(proj[:num, 0], proj[:num, 1], c=color[predict_reg[:]],marker='o', s=40)
    ax.scatter(proj[num:num2, 0], proj[num:num2, 1] ,c=color[predict_noi[:]],marker='o', s=40)
    ax.scatter(proj[num2:, 0], proj[num2:, 1], c=color[predict_pert[:]],marker='^', s=250)

  if save:
      plt.savefig(where+'.png')




def isomap_video(base_network, img_reg, img_noise, img_pert, num, vert=True, save=False, where=False):
    
  ind = np.random.randint(0,len(img_reg),num)
  num2 = 2 * num
    
  base_network.to('cpu')
  embedding_reg, logits_reg = base_network(img_reg[ind])
  embedding_noise, logits_noise = base_network(img_noise[ind])
  embedding_pert, logits_pert = base_network(img_pert)

  total = []
  total.append(embedding_reg)
  total.append(embedding_noise)
  total.append(embedding_pert)

  total = torch.cat(total)
  len(total)

  softmax_reg = nn.Softmax(dim=1)(1.0 * logits_reg)
  _, predict_reg = torch.max(softmax_reg, 1)

  softmax_noi = nn.Softmax(dim=1)(1.0 * logits_noise)
  _, predict_noi = torch.max(softmax_noi, 1)

  softmax_pert = nn.Softmax(dim=1)(1.0 * logits_pert)
  _, predict_pert = torch.max(softmax_pert, 1)


  predict_reg = predict_reg.tolist()
  predict_noi = predict_noi.tolist()
  predict_pert = predict_pert.tolist()

  color = np.array(['r', 'b', 'k'])

  X = np.arange(-5, 5, 0.25)
  Y = np.arange(-5, 5, 0.25)
  X, Y = np.meshgrid(X, Y)
  R = np.sqrt(X**2 + Y**2)
  Z = np.sin(R)

  fig = plt.figure(figsize = (20, 10))
  ax = plt.gca()
  ax = plt.axes(projection='3d')
  plt.title('Isomap 3D \n Curves: Y10 (orange), Y1 noisy (blue) \n Point color - predicted class: spiral (red), elliptical (blue), merger (black) \n Perturbed images - large points', fontsize=25)

  def init():
    model = Isomap(n_components=3)
    proj = model.fit_transform(total.detach().numpy())

    ax.plot_trisurf(proj[:num, 0], proj[:num, 1], proj[:num, 2], cmap="autumn", alpha = 0.3)
    ax.plot_trisurf(proj[num:num2, 0], proj[num:num2, 1], proj[num:num2, 2],cmap="winter", alpha=0.3)
    ax.plot_trisurf(proj[num2:, 0], proj[num2:, 1], proj[num2:, 2],cmap="Reds", alpha=0.3)

    ax.scatter(proj[:num, 0], proj[:num, 1], proj[:num, 2], c=color[predict_reg[:]],marker='o', s=40)
    ax.scatter(proj[num:num2, 0], proj[num:num2, 1], proj[num:num2, 2],c=color[predict_noi[:]],marker='o', s=40)
    ax.scatter(proj[num2:, 0], proj[num2:, 1], proj[num2:, 2],c=color[predict_pert[:]],marker='^', s=250)
    return fig,

  if vert:
    def animate(i):
      # elevation angle : -180 deg to 180 deg
      ax.view_init(elev=(i-45)*4, azim=10)
      return fig,
  else:
    def animate(i):
      # azimuth angle : 0 deg to 360 deg
      ax.view_init(elev=10, azim=i*4)
      return fig,

  # Animate
  ani = animation.FuncAnimation(fig, animate, init_func=init,
                            frames=90, interval=50, blit=True)

  if save:
    fn = where
    ani.save(fn+'.mp4',writer='ffmpeg',fps=1000/50)
  
  return ani
