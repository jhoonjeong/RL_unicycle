import DDPG
import tensorflow as tf
import time


config = tf.ConfigProto()  # Allocated memory of GPU grow
config.gpu_options.allow_growth = True


def main(video_name = None):
    s_size = 19
    with tf.device('/cpu'):
        with tf.Session() as session:
            actor = DDPG.Actor(session, a_bound=1., s_size=s_size, a_size=2)
            critic = DDPG.Critic(session, s_size=s_size, a_size=2)

            saver = tf.train.Saver()
            save_path = saver.restore(session, "data/tf/DDPG.ckpt")
            print("model restored")

            for i in range(3):
                s = time.time()
                env = DDPG.Environment(render=True, sigma=0.02, down=1.0, get_image=False)

                if video_name:
                    env.record("data/mov/" + video_name + ".mp4")

                step = env.replay(actor.policy_one)

                print("unicycle lasted {} steps and {:2f} seconds.".format(step, step / 30))
                print("time = {}".format(time.time() - s))
                env.close()

    end = time.time()

if __name__ == '__main__':
    main(video_name="DDPG")



