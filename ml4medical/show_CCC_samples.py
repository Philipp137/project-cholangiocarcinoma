from ml4medical.dataset import DataModule
from ml4medical.visualize import show_transforms
# data_folder='D:/Arbeit/cholangiocarcinoma-data/CRC_KR/'
# variant = 'MSIMSS'
data_folder='/work/nb671233/data/CCC_01/CCC/'
variant = 'CCC'
data_module = DataModule(root_folder=data_folder, data_variant=variant, train_batch_size=2,
                         val_batch_size=2, train_subbatch_size=0, val_return_slide_number=True,
                         val_subbatch_size=0, normalize=None, distributed=False, num_workers=2, persistent_workers=False)

show_transforms(data_module, 100, 10, variant=variant, case='train')
show_transforms(data_module, 100, 10, variant=variant, case='val')
