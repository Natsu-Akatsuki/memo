with open("test_pcd_binary.pcd", 'r') as f:  # orbita_pcd  test_pcd_binary
    data = f.readlines()
    pass


def load_pcd_data(file_path, remove_nan=True):
    """
    pcd点云 -> numpy
    Args:
        remove_nan:
        file_path:
    """

    with open(file_path, 'r') as f:
        data = f.readlines()

    # note: 这种字符型的可以直接由loadtxt读取，不需要逐行遍历，但实测速度很慢
    # pointcloud = np.loadtxt(data[13:], dtype=np.float32)
    # todo: 目前数据段的识别需人工识别和修正，可更自动化

    return pointcloud
