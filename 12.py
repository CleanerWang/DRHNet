# -*- coding: utf-8 -*-
"""
---------------------------------------------------------
   file Name:
   Description:
   Author:
   date:
---------------------------------------------------------
 change Activity:
---------------------------------------------------------
"""

import  cv2
import numpy as np
import torch
import numpy

a = cv2.imread("F:\code\comparison algorithm\DRHNet-master\DRHNet-master\code\samples\\1.jpg")
a = torch.from_numpy(a)
a = a.to(torch.float32)
b = 255.0*torch.ones_like(a)
c = torch.min(a,b)
print(a,c)
