import imageio
imageio.plugins.ffmpeg.download()
import moviepy.editor as ed


def videoToGif(video, gif_name):

    clip = (ed.VideoFileClip(video)
            # start and end time. format: (minutes, seconds.miliseconds?)
            .subclip((0, 1.0), (0, 12.0))
            .resize(0.3))
    clip.write_gif(gif_name, fps=25)
    return clip
