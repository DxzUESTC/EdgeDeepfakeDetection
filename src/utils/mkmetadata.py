import os

'''
    这个脚本用来创建存放原始视频数据的视频索引和标注信息
    脚本会在项目根目录下的'./data/datasets/metadata'目录下创建csv文件
    csv文件包含：
    video_id: 视频的唯一标识符
    video_path: 视频文件的路径
    label: 视频的标签信息
    split: 视频的分割信息（如训练集、验证集、测试集）
'''

def create_metadata():
    metadata_dir = 'D:\\09_Project\\BWTAC2025\\EdgeDeepfakeDetection\\data\\datasets\\metadata'
    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)
    metadata_file = os.path.join(metadata_dir, 'video_metadata_c23.csv')
    with open(metadata_file, 'w') as f:
        f.write('video_id,video_path,label,split\n')
        
        # 添加original video metadata
        # 输入包含original视频数据的目录，读取目录下的视频文件名
        # video_id命名为0000, 0001, ... id的格式为4位数字，左起第一位数字固定为0表示为original视频数据，剩下数字为源视频文件名
        # video_path为视频文件的相对路径
        # label都为 0 表示真实视频数据
        # split 按照随机顺序分为训练集，验证集，测试集，比例为7:1.5:1.5，分别表示为0, 1, 2
        
        original_videos_dir = "D:\\09_Project\\BWTAC2025\\EdgeDeepfakeDetection\\data\\datasets\\original\\c23"
        video_files = os.listdir(original_videos_dir)
        video_files.sort()  # 确保视频文件按名称排序
        for idx, video_file in enumerate(video_files):
            video_id = f'0{idx:03d}'  # 生成4位数字的video_id
            video_path = os.path.join(original_videos_dir, video_file)
            label = 0  # 所有original视频的标签为0
            if idx < len(video_files) * 0.7:
                split = 0  # 70%为训练集
            elif idx < len(video_files) * 0.85:
                split = 1  # 15%为验证集
            else:
                split = 2  # 15%为测试集
            f.write(f'{video_id},{video_path},{label},{split}\n')
        
        # 添加deepfakes video metadata
        # 输入包含deepfakes视频数据的目录，读取目录下的视频文件名
        # video_id命名为1001, 1002, ... id的格式为4位数字，左起第一位数字固定为1表示为deepfakes视频数据，剩下数字为源视频文件名前3位数字
        # video_path为视频文件的相对路径
        # label都为 1 表示伪造的视频
        # split 按照真实视频的划分分为训练集，验证集，测试集，比例为7:1.5:1.5，分别表示为0, 1, 2
        
        deepfakes_videos_dir = "D:\\09_Project\\BWTAC2025\\EdgeDeepfakeDetection\\data\\datasets\\deepfakes\\c23"
        deepfake_files = os.listdir(deepfakes_videos_dir)
        deepfake_files.sort()  # 确保deepfake文件按名称排序
        for idx, video_file in enumerate(deepfake_files):
            video_id = f'1{idx:03d}'  # 生成4位数字的video_id
            video_path = os.path.join(deepfakes_videos_dir, video_file)
            label = 1  # 所有deepfakes视频的标签为1
            if idx < len(deepfake_files) * 0.7:
                split = 0  # 70%为训练集
            elif idx < len(deepfake_files) * 0.85:
                split = 1  # 15%为验证集
            else:
                split = 2  # 15%为测试集
            f.write(f'{video_id},{video_path},{label},{split}\n')
        
        # 添加Face2Face video metadata
        # 输入包含Face2Face视频数据的目录，读取目录下的视频文件名
        # video_id命名为2001, 2002, ... id的格式为4位数字，左起第一位数字固定为2表示为Face2Face视频数据，剩下数字为源视频文件名前3位数字
        # video_path为视频文件的相对路径
        # label都为 1 表示伪造的视频
        # split 按照真实视频的划分分为训练集，验证集，测试集，比例为7:1.5:1.5，分别表示为0, 1, 2

        Face2Face_videos_dir = "D:\\09_Project\\BWTAC2025\\EdgeDeepfakeDetection\\data\\datasets\\Face2Face\\c23"
        Face2Face_videos_files = os.listdir(Face2Face_videos_dir)
        Face2Face_videos_files.sort()  # 确保Face2Face文件按名称排序
        for idx, video_file in enumerate(Face2Face_videos_files):
            video_id = f'2{idx:03d}'  # 生成4位数字的video_id
            video_path = os.path.join(Face2Face_videos_dir, video_file)
            label = 1  # 所有Face2Face视频的标签为1
            if idx < len(Face2Face_videos_files) * 0.7:
                split = 0  # 70%为训练集
            elif idx < len(Face2Face_videos_files) * 0.85:
                split = 1  # 15%为验证集
            else:
                split = 2  # 15%为测试集
            f.write(f'{video_id},{video_path},{label},{split}\n')

        # 添加FaceShifter video metadata
        # 输入包含FaceShifter视频数据的目录，读取目录下的视频文件名
        # video_id命名为3001, 3002, ... id的格式为4位数字，左起第一位数字固定为3表示为FaceShifter视频数据，剩下数字为源视频文件名前3位数字
        # video_path为视频文件的相对路径
        # label都为 1 表示伪造的视频
        # split 按照真实视频的划分分为训练集，验证集，测试集，比例为7:1.5:1.5，分别表示为0, 1, 2

        FaceShifter_videos_dir = "D:\\09_Project\\BWTAC2025\\EdgeDeepfakeDetection\\data\\datasets\\FaceShifter\\c23"
        FaceShifter_videos_files = os.listdir(FaceShifter_videos_dir)
        FaceShifter_videos_files.sort()  # 确保FaceShifter文件按名称排序
        for idx, video_file in enumerate(FaceShifter_videos_files):
            video_id = f'3{idx:03d}'  # 生成4位数字的video_id
            video_path = os.path.join(FaceShifter_videos_dir, video_file)
            label = 1  # 所有FaceShifter视频的标签为1
            if idx < len(FaceShifter_videos_files) * 0.7:
                split = 0  # 70%为训练集
            elif idx < len(FaceShifter_videos_files) * 0.85:
                split = 1  # 15%为验证集
            else:
                split = 2  # 15%为测试集
            f.write(f'{video_id},{video_path},{label},{split}\n')

        # 添加FaceSwap video metadata
        # 输入包含FaceSwap视频数据的目录，读取目录下的视频文件名
        # video_id命名为4001, 4002, ... id的格式为4位数字，左起第一位数字固定为4表示为FaceSwap视频数据，剩下数字为源视频文件名前3位数字
        # video_path为视频文件的相对路径
        # label都为 1 表示伪造的视频
        # split 按照真实视频的划分分为训练集，验证集，测试集，比例为7:1.5:1.5，分别表示为0, 1, 2

        FaceSwap_videos_dir = "D:\\09_Project\\BWTAC2025\\EdgeDeepfakeDetection\\data\\datasets\\FaceSwap\\c23"
        FaceSwap_videos_files = os.listdir(FaceSwap_videos_dir)
        FaceSwap_videos_files.sort()  # 确保FaceSwap文件按名称排序
        for idx, video_file in enumerate(FaceSwap_videos_files):
            video_id = f'4{idx:03d}'  # 生成4位数字的video_id
            video_path = os.path.join(FaceSwap_videos_dir, video_file)
            label = 1  # 所有FaceSwap视频的标签为1
            if idx < len(FaceSwap_videos_files) * 0.7:
                split = 0  # 70%为训练集
            elif idx < len(FaceSwap_videos_files) * 0.85:
                split = 1  # 15%为验证集
            else:
                split = 2  # 15%为测试集
            f.write(f'{video_id},{video_path},{label},{split}\n')

        # 添加NeuralTextures video metadata
        # 输入包含NeuralTextures视频数据的目录，读取目录下的视频文件名
        # video_id命名为5001, 5002, ... id的格式为4位数字，左起第一位数字固定为5表示为NeuralTextures视频数据，剩下数字为源视频文件名前3位数字
        # video_path为视频文件的相对路径
        # label都为 1 表示伪造的视频
        # split 按照真实视频的划分分为训练集，验证集，测试集，比例为7:1.5:1.5，分别表示为0, 1, 2

        NeuralTextures_videos_dir = "D:\\09_Project\\BWTAC2025\\EdgeDeepfakeDetection\\data\\datasets\\NeuralTexture\\c23"
        NeuralTexture_videos_files = os.listdir(NeuralTextures_videos_dir)
        NeuralTexture_videos_files.sort()  # 确保NeuralTexture文件按名称排序
        for idx, video_file in enumerate(NeuralTexture_videos_files):
            video_id = f'5{idx:03d}'  # 生成4位数字的video_id
            video_path = os.path.join(NeuralTextures_videos_dir, video_file)
            label = 1  # 所有NeuralTexture视频的标签为1
            if idx < len(NeuralTexture_videos_files) * 0.7:
                split = 0  # 70%为训练集
            elif idx < len(NeuralTexture_videos_files) * 0.85:
                split = 1  # 15%为验证集
            else:
                split = 2  # 15%为测试集
            f.write(f'{video_id},{video_path},{label},{split}\n')

        print(f'Metadata file created at {metadata_file}')


if __name__ == '__main__':
    create_metadata()   
# 运行脚本以创建metadata文件
# 运行脚本以创建metadata文件
# python mkmetadate.py
# 运行后会在./data/datasets/metadata/c23目录下生成video_metadata.csv文件