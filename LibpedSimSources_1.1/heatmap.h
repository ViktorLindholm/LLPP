#pragma once

void updateheatmapCUDA(int *d_w, int *d_blurred_heatmap, int *d_heatmap, int *d_scaled_heatmap, int* heatmap, int* scaled_heatmap, int* blurred_heatmap, int *xDesired, int *yDesired, int number_of_agents);