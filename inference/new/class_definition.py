import cv2
from torch import nn


class Contour_Checking_fn(object):
	# Defining __call__ method
	def __call__(self, pt):
		raise NotImplementedError

# Easy version of 4pt contour checking function - 1 of 4 points need to be in the contour for test to pass
class isInContourV3_Easy(Contour_Checking_fn):
  def __init__(self, contour, patch_size, center_shift=0.5):
    self.cont = contour
    self.patch_size = patch_size
    self.shift = int(patch_size // 2 * center_shift)

  def __call__(self, pt):
    center = (pt[0] + self.patch_size // 2, pt[1] + self.patch_size // 2)
    if self.shift > 0:
      all_points = [(center[0] - self.shift, center[1] - self.shift),
                    (center[0] + self.shift, center[1] + self.shift),
                    (center[0] + self.shift, center[1] - self.shift),
                    (center[0] - self.shift, center[1] + self.shift)
                    ]
    else:
      all_points = [center]

    for points in all_points:
      if cv2.pointPolygonTest(self.cont, (int(points[0]), int(points[1])), False) >= 0:
        return 1
    return 0
    
class MLP(nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(401, 100),
      nn.ReLU(),
      nn.Linear(100, 50),
      nn.ReLU(),
      nn.Linear(50, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)