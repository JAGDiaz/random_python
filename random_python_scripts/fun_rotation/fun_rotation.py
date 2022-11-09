import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anime

def rotate_3d(X, Y, Z, angle_x=np.pi*.5, angle_y=np.pi*.5, angle_z=np.pi*.5):
    vectors = np.array([X, Y, Z])

    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0], 
                   [0, 0, 1]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0,1,0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rx = np.array([[1,0,0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])

    R = Rx@Ry@Rz

    return np.einsum('ij,jkl -> ikl', R, vectors)

def rotate_about_Z(X, Y, Z, angle=np.pi*.5):
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0], 
                   [0, 0, 1]])
    
    vectors = np.array([X,Y,Z])
    return np.einsum('ij,jkl -> ikl', R, vectors)

def rotate_surface(x,y,func,t=0):
    R = np.linalg.norm([x,y], ord=2, axis=0)
    T = np.arctan2(y,x)
    T+=t 
    x_new = R*np.cos(T)
    y_new = R*np.sin(T)

    return func(x_new, y_new)
    
def rot_surf(x, y, t, func):
    R = np.linalg.norm([x,y], ord=2, axis=0)
    T = np.arctan2(y,x)
    T+=t 
    x_new = R*np.cos(T)
    y_new = R*np.sin(T)

    return func(x_new, y_new)

def u(x,y):
    return np.where(np.abs(2*x) + np.abs(y) < 1, 1 - np.abs(2*x) - np.abs(y), 0)

def u2(x,y):
    return np.where(2*x**2 + y**2 < 1, np.sqrt(1 - 2*x**2 - y**2), 0)

def u3(x,y):
    return x + y

meta = dict(title=f"YOUSPINMERIGHTROUND", 
            artist="Matplotlib")
frame_number = 100
seconds = 5
writer = anime.FFMpegWriter(fps=frame_number//seconds, metadata=meta)
times = np.linspace(0,2*np.pi,frame_number)

x, y = np.linspace(-2.5, 2.5, 1001), np.linspace(-2.5, 2.5, 1001)

X, Y, T = np.meshgrid(x,y,times)

Z = rot_surf(X,Y,T,u3)

X = np.transpose(X, axes=[2,0,1])
Y = np.transpose(Y, axes=[2,0,1])
Z = np.transpose(Z, axes=[2,0,1])

fig = plt.figure(figsize=(10,7))

ax = fig.add_subplot(111, projection='3d')
ax.set(xlim=x[[0,-1]], ylim=y[[0,-1]], zlim=(Z.min(), Z.max()))
ax.set_title(f"YOU SPIN ME RIGHT ROUND, BABY, RIGHT ROUND. {666:3d}$^o$")

surf = ax.plot_surface(X[0], Y[0], Z[0], color='tab:red')
fig.tight_layout()

with writer.saving(fig, "RIGHT_ROUND.gif", 150):

    for ii in range(frame_number):
        surf.remove()
        #surf = ax.plot_surface(X,Y,rotate_surface(X,Y,u,times[ii]), color='tab:red')
        surf = ax.plot_surface(X[ii],Y[ii], Z[ii], color='tab:red')
        ax.set_title(f"YOU SPIN ME RIGHT ROUND, BABY, RIGHT ROUND. {int(360*(ii+1)/frame_number):3d}$^o$")

        #X, Y, Z = rotate_3d(X, Y, Z, angle_add, angle_add, angle_add)

        writer.grab_frame()
        print(f"Frame {ii+1:3d}/{frame_number} done!")

plt.close(fig)
