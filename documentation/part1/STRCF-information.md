# An Overview of Spatially Regularized Discriminative Correlation Filters (STRCFs)

## Approach Summary

STRCFs are based on Discriminative Correlation Filters (DCF), which have emerged as a powerful tool for visual tracking due to their efficiency and robustness. DCF-based trackers learn a correlation filter that makes a peak in the response map at the target location. The filter is learned by minimizing the error between the desired response and the actual response of the filter on training samples. Even though they are effective and have been transformational, DCF methods can suffer from boundary effects which lead to inaccurate tracking, especially near the edge of the tracking area. This is due DCFs using circular correlation, which introduces artificial boundaries in the training and detection samples. More background information on DCFs is available in my review of _Efficient and Practical Correlation Filter Tracking_ in our [research notes](/documentation/research_notes.pdf)

## Resources

- [Paper](https://arxiv.org/pdf/1803.08679)
- [GitHub Repo](https://github.com/lifeng9472/STRCF)

## Strengths & Weaknesses

STRCF aims to solve the boundary effects of traditional DCF-based trackers by incorporating both spatial and temporal regularization. Spatial regularization penalizes filter coefficients based on location, allowing the tracker to learn from larger image regions without corrupting the target samples. Temporal regularization, influenced by the Passive-Aggressive algorithm, ensures smooth adaptation of the correlation filter over time. These combined strategies help mitigate edge artifacts, keep predictions consistent during quick target changes, and provide a more efficient approximation of SRDCF

STRCF excels in scenarios with occlusion, fast motion, deformations, scale variation, and background clutter. Its temporal regularization helps greatly with partial or full occlusion, as it can re-acquire the target when it reappears. Spatial regularization further helps with handling deformations and scale changes by enabling the filter to adapt. Because of this, STRCF consistently outperforms KCF and shows marked gains on multiple benchmarks.

However, STRCF ie computationally more expensive than simpler trackers such as KCF. Its performance can also be sensitive to parameter choices, which may require careful tuning. Potential improvements include developing adaptive regularization mechanisms, incorporating deep learning features for richer representations, and integrating second-order data-fitting for better handling of complex appearance variations [(Li et al.)](https://arxiv.org/pdf/1803.08679).

## Ensemble for Single Object Tracking

- STRCF will perform much better if motion is very fast. We could potentially use an optical flow algorithm, or some kind of extra module on top of the ensemble model that could estimate the objects speed, and give more weight to STRCF in these scenarios.
- If you detect partial or full occlusion (like a sudden dip in response from one tracker, or external occlusion cues) or the object is moving near the frame boundary, give higher weight to STRCF, since its spatial and temporal regularization are best for these conditions.

## Appendix: Deep Learning

STRCF supports various feature types, such as: HOG, color features, and notably CNN features. The CNN features can be enabled through a module, and even though they are from a pre-trained network, this still provides us the option to extend into deep learning.
