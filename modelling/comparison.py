import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def find_difference(output_dict, cluster, index, plot=False):
    """
    baya önemli bir şey bence bugün yaptım

    hiç cluster edilmemiş ile cluster edilmiş verileri, cluster bazında karşılaştırır.

    """
    one_cluster_error = output_dict["cluster_1"]["inside_cluster"]["cluster_1"]["error_values"]
    current_cluster_error = output_dict[cluster]["inside_cluster"]["cluster_" + str(index)]["error_values"]
    cluster_labels = output_dict[cluster]["test_labels"]
    cluster_index = np.where(cluster_labels == index - 1)
    one_cluster_error_loc = one_cluster_error[cluster_index]
    if plot == True:
        f, ax = plt.subplots(1, 2)
        f.set_figheight(15)
        f.set_figwidth(30)
        sns.histplot(data=current_cluster_error, kde=True, ax=ax[0])
        sns.histplot(data=one_cluster_error_loc, kde=True, ax=ax[1])
        ax[0].set_title("CLUSTERED VERSION", fontsize=30)
        ax[1].set_title("NON-CLUSTERED VERSION", fontsize=30)
        plt.suptitle(cluster + " " + str(index), fontsize=30)
        plt.show()
    return one_cluster_error_loc, current_cluster_error