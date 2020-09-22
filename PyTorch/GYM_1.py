import gym
env  = gym.make('CartPole-v0')
env.reset()
for i in range(100000):
    env.render() #显示
    env.step(1)

    env.step(0)
    env.step(0)
    env.step(1)
    env.step(1)
    env.step(0)
    env.step(0)
    env.step(1)

    s,r,done,info = env.step(0)  #输入动作0/1，返回S,A,done,info
    #done : 游戏是否完成
    #info:用于诊断和调试
    if i %1000==0:
        print(s,r,done,info)

