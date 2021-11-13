import matplotlib.pyplot as plt
from detectors.mc import make_blob


def test_generates():
    blob = make_blob(class_id=1, h=400, w=400)
    plt.imshow(blob)
    plt.show()
