from ..models.visual_attributes import VisualAttrConfig

visual_attribute_config = [
    VisualAttrConfig('L', 0.5, 0.01, mean=56.16887621, std=3.416602627),
    VisualAttrConfig('a', 0.5, 0.0001, mean=-13.3509608, std=4.958611063),
    VisualAttrConfig('b', 0.5, 0.0001, mean=42.68991623, std=4.907432068),
    # VisualAttrConfig('contrast', 0.4, 0.0001, mean=130.8824943, std=49.53392233),
    # VisualAttrConfig('correlation', 0.1, 0.0001, mean=0.983804451, std=0.006855397),
    # VisualAttrConfig('energy', 0.5, 0.0001, mean=0.249356799, std=0.015899871),
    # VisualAttrConfig('entropy', 0.5, 0.001, mean=6.006077151, std=0.155426234),
    VisualAttrConfig('homogeneity', 0.5, 0.0001, mean=0.547540533, std=0.066137278),
    # VisualAttrConfig('uniformity', 0.5, 0.0001, mean=0.065048235, std=0.00932834),
    VisualAttrConfig('equivalent_diameter', 0.5, 0.001, mean=10.76372478, std=2.24162268),
    # VisualAttrConfig('eccentricity', 0.5, 0.0001, mean=0.548399747, std=0.070028959),
    VisualAttrConfig('feret_diameter_max', 0.5, 0.001, mean=12.65424064, std=2.836734251),
    VisualAttrConfig('filled_area', 0.5, 0.001, mean=101.5903847, std=45.4936738),
    VisualAttrConfig('perimeter', 0.5, 0.01, mean=39.84774916, std=9.143521734),
    # VisualAttrConfig('roundness', 0.5, 0.01, mean=0.729736846, std=0.0604815),
        ]