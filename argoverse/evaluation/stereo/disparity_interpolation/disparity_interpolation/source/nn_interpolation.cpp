#include "nn_interpolation.h"
#include <stdint.h>

// interpolate all missing (=invalid) disparities
cv::Mat nn_interpolation(cv::Mat disparity)
{
    int height = disparity.rows;
    int width = disparity.cols;

    // for each row do
    for (int v = 0; v < height; v++) {
      // init counter
      int count = 0;
      // for each pixel do
      for (int u = 0; u < width; u++) {
        // if disparity valid
        if (disparity.at<float>(v, u) >= 0) {
          // at least one pixel requires interpolation
          if (count >= 1) {
            // first and last value for interpolation
            int u1 = u - count;
            int u2 = u - 1;

            // set pixel to min disparity
            if (u1 > 0 && u2 < width - 1) {
              float d_ipol = std::min(disparity.at<float>(v, u1 - 1), disparity.at<float>(v, u2 + 1));
              for (int u_curr = u1; u_curr <= u2; u_curr++)
                disparity.at<float>(v, u_curr) = d_ipol;
            }
          }
          // reset counter
          count = 0;
        // otherwise increment counter
        } else {
          count++;
        }
      }

      // extrapolate to the left
      for (int u = 0; u < width; u++) {
        if (disparity.at<float>(v, u) >= 0) {
          for (int u2 = 0; u2 < u; u2++)
            disparity.at<float>(v, u2) = disparity.at<float>(v, u);
          break;
        }
      }

      // extrapolate to the right
      for (int u = width - 1; u >= 0; u--) {
        if (disparity.at<float>(v, u) >= 0) {
          for (int u2 = u + 1; u2 <= width - 1; u2++)
            disparity.at<float>(v, u2) = disparity.at<float>(v, u);
          break;
        }
      }
    }

    // for each column do
    for (int u = 0; u < width; u++) {
      // extrapolate to the top
      for (int v = 0; v < height; v++) {
        if (disparity.at<float>(v, u) >= 0) {
          for (int v2 = 0; v2 < v; v2++)
            disparity.at<float>(v2, u) = disparity.at<float>(v, u);
          break;
        }
      }

      // extrapolate to the bottom
      for (int v = height - 1; v >= 0; v--) {
        if (disparity.at<float>(v, u) >= 0) {
          for (int v2 = v+1; v2 <= height - 1; v2++)
            disparity.at<float>(v2, u) = disparity.at<float>(v, u);
          break;
        }
      }
    }

    return disparity;
}
