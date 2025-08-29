import os
import cv2
""""
    这个脚本用来对视频数据集进行抽帧采样
    采样为单个视频抽取固定数量的帧
    因为训练所用的真实视频数量远小于基于真实视频制作的伪造视频数量
    所以单个真实视频抽取固定帧数为50帧，每一种伪造方法对应的单个伪造视频抽取固定帧数为10帧
    总的来说，真实视频最后的帧数与伪造视频最后的帧数比例为1:1
"""
def extract_frames(video_path, output_dir, num_frames, output_format='png'):
    """
    :param video_path: 视频文件路径
    :param output_dir: 输出目录
    :param num_frame: 要抽取的帧数
    :param output_format: 输出帧的格式
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频文件名（不带扩展名）
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算帧间隔（均匀分布）
    if num_frames < 1:
        raise ValueError("目标帧数必须大于 0")
    if num_frames > total_frames:
        num_frames = total_frames  # 如果目标帧数超过总帧数，限制为总帧数
    
    # 帧间隔（整数），每隔多少帧取一帧
    frame_interval = total_frames // num_frames if num_frames > 1 else 1
    selected_indices = [i * frame_interval for i in range(num_frames)]
    
    # 调整最后一个索引，确保不超过总帧数
    if num_frames > 1 and selected_indices[-1] >= total_frames:
        selected_indices[-1] = total_frames - 1
    
    # 抽取帧
    for idx, frame_idx in enumerate(selected_indices):
        # 设置视频帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"无法读取帧 {frame_idx}")
            continue
        
        # 生成输出文件名，格式为 frame_XXXX.jpg
        output_path = os.path.join(output_dir, f"{video_name}_frame_{idx + 1:04d}.{output_format}")
        
        # 保存帧为图片
        cv2.imwrite(output_path, frame)
        print(f"保存帧 {frame_idx} 到 {output_path}")
    
    # 释放视频对象
    cap.release()
    print(f"完成！抽取了 {len(selected_indices)} 帧，保存到 {output_dir}")

# 从视频集中提取帧
def extract_from_original():
    """
    :param original_videos_dir: 原始视频目录
    :param output_dir: 输出目录
    :param num_frames: 每个视频抽取的帧数
    """
    original_videos_dir = "E:\\DataSet\\FaceForensics\\temp"
    output_dir = "E:\\DataSet\\FaceForensics\\preprocessed\\dataset_frames\\deepfakes\\c23"
    video_files = [f for f in os.listdir(original_videos_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    num_frames = 10  # 每个真实视频抽取50帧
    
    for video_file in video_files:
        video_path = os.path.join(original_videos_dir, video_file)
        extract_frames(video_path, output_dir, num_frames)


if __name__ == "__main__":
    # 从原始视频集中提取帧
    extract_from_original()