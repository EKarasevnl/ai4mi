import unittest

import torch

from vit_unet import ViTSegmenter


class ViTSegmenterForwardTest(unittest.TestCase):
    def test_forward_shape(self) -> None:
        model = ViTSegmenter(in_channels=1, num_classes=5)
        model.init_weights()
        model.eval()

        input_tensor = torch.randn(2, 1, 256, 256)

        with torch.no_grad():
            output = model(input_tensor)

        self.assertEqual(output.shape, (2, 5, 256, 256))
        self.assertFalse(torch.isnan(output).any().item())


if __name__ == "__main__":
    unittest.main()
