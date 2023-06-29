import ffmpeg
from PIL import Image
from src.envs import ButtonFood

env = ButtonFood(
    render_mode='human'
)

imgs = []
obs, info = env.reset(init_agent=(0.5, 0.5), init_button=(0.5, 0.9), init_target=(0.8, 0.3))

for i in range(50):
    if i < 12: action = (0.00, +0.03)
    else:      action = (0.01, -0.02)

    obs, r, done, info = env.step(action)

    frame = env.render(render_mode = 'rgb_array')

    imgs.append(Image.fromarray(frame))

    # if done:
    #     break

env.close()

for f, img in enumerate(imgs):
    img.save(f'frames/render_example_{str(f).zfill(2)}.jpg')

# imgs[0].save('render_example.gif', save_all=True, append_images=imgs[1:], fps=8, loop=0)

(
    ffmpeg
    .input('frames/*.jpg', pattern_type='glob', framerate=10)
    .output('movie.mp4')
    .run()
)
