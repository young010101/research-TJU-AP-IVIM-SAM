import nibabel as nib
import h5py

# NIfTI文件路径
nii_file_path = '/data/users/cyang/acute_pancreatitis/unprocess/nii/pantient2/ST0_RTrAx_T2_fs_2mm_20231120173431_7.nii.gz'

# HDF5文件路径
h5_file_path = 'path_to_your_output_h5_file.h5'

# 使用nibabel读取NIfTI文件
nii_img = nib.load(nii_file_path)

# 获取NIfTI文件中的图像数据
nii_data = nii_img.get_fdata()

print(nii_data.shape, nii_data.dtype, type(nii_data))
print(nii_data.min(), nii_data.max())
print(nii_img.header,nii_img.header_class,nii_img.header.extensions)
print(type(nii_img.header))

# 打开一个新的HDF5文件，准备写入
with h5py.File(h5_file_path, 'w') as h5_file:
    # 创建一个与NIfTI数据形状相匹配的数据集
    h5_dataset = h5_file.create_dataset('nii_data', data=nii_data)

    # 你还可以添加一些元数据，例如NIfTI文件的头部信息
    # h5_file['header'] = nii_img.header.tolist()