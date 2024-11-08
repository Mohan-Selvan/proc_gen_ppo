import matplotlib.pyplot as plt
from IPython import display

plt.ion()

def plot(scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    #plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
    #plt.tight_layout()
    plt.savefig("./plot.png")

def lerp(a, b, t):
    return (a + ((b-a) * t))

def lerp_color(from_color, to_color, t):
    return (
        lerp(from_color[0], to_color[0], t),
        lerp(from_color[1], to_color[1], t),
        lerp(from_color[2], to_color[2], t)
    )