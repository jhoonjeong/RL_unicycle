import DQN
import tensorflow as tf
import time

config = tf.ConfigProto()  # Allocated memory of GPU grow
config.gpu_options.allow_growth = True


def main(weights_name, video_name=None, get_image=False):
    env = DQN.Environment(render=True, sigma=0.02, down=1.0, get_image=get_image)
    s_size = env.env.s_size

    agent = DQN.Agent(s_size=s_size)
    agent.network.model.load_weights("data/" + weights_name + ".h5", by_name=True)
    print("model loaded")

    for _ in range(3):
        s = time.time()

        if video_name:
            env.record("data/mov/" + video_name + ".mp4")

        step = env.replay(agent.policy)
        print("unicycle lasted {} steps and {:2f} seconds.".format(step, step/30))
        print("time = {}".format(time.time() - s))
    env.close()


if __name__ == '__main__':
    main("unicycle_DQN", video_name="DQN")
    # main("file_name", get_image=True)