# YeastPed: Automated Genealogy Tracking of Budding Yeast from Timelapse Microscopy Images - Without a Fluorescent Budneck Marker

This program was designed to aid in cell image analysis of budding yeast. The program incorporates four important functionalities,
which have been optimized for budding yeast, into an automated workflow that can be run via the command line interface:
1)	Segmentation
2)	Artifact Removal
3)	Frame-to-Frame Tracking
4)	Genealogy reconstruction

This pipeline incorporates both previously developed algorithms and a novel genealogy reconstruction method.
A convolutional neural network (CNN) trained on segmented images of budding yeast  (Dietler et al., 2020) performs the segmentation, 
and our artifact removal method utilizes a minimum life span and minimum cell area criteria. Tracking is based upon Lineage Mapper –
a Hungarian algorithm-based tool optimized for cell tracking (Chalfoun et al., 2016). The final step, genealogy reconstruction,
is based upon assumptions of mother-daughter proximity and the observation that the major axis of elliptical yeast daughter cells align
with possible mother cells. The geneaology tracking does not require a flourescent budneck marker.

Based on benchmark of the genealogy’ accuracy on three 200 image datasets, this novel algorithm correctly identified mother-daughter pairs
88.3% of the time.

Please refer to the Doc.pdf file in the /doc folder. There, you can find detailed usage instructions and more details about the functionalites provided by YeastPed.

Citations:

Chalfoun, J., Majurski, M., Dima, A., Halter, M., Bhadriraju, K., & Brady, M. (2016). Lineage mapper:
  A versatile cell and particle tracker. Scientific Reports, 6(1), 36984. https://doi.org/10.1038/srep36984

Dietler, N., Minder, M., Gligorovski, V., Economou, A. M., Joly, D. A. H. L., Sadeghi, A., Chan, C. H. M., Koziński,
  M., Weigert, M., Bitbol, A.-F., & Rahi, S. J. (2020). A convolutional neural network segments yeast microscopy images
  with high accuracy. Nature Communications, 11(1), 5723. https://doi.org/10.1038/s41467-020-19557-4
