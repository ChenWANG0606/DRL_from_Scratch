import gym
from matplotlib import animation
import matplotlib.pyplot as plt
 
env = gym.make('CartPole-v1', render_mode='rgb_array')
obs, _ = env.reset()
 
kp = 0.000
kv = -0.002
ka = -0.3
kav = -0.01
ks = -0.000
sum_angle = 0.000
frames = []
 
def save_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=5)
    anim.save('./logs/CartPortCtrl.gif', writer='imagemagick', fps=30)
 
def CalcAction(obs):
    action = 0 # 0 meanleft, 1 means right
    global sum_angle
    sum = kp * obs[0] + kv * obs[1] + ka * obs[2] + kav * obs[3] + ks * sum_angle
    sum_angle += obs[2]
    if (sum < 0.0):
        action = 1
    else:
        action = 0
    return action
 
for _ in range(500):
    frames.append(env.render())
    action = CalcAction(obs)
    print('action = %d' % action)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
save_gif(frames)
 
 