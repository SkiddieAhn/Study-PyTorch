from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
import torch.nn.functional as F

'''
# TensorBoard 확인법 
# 콘솔창에 아래 명령어 입력
# ex) tensorboard --logdir=runs/fashion_mnist_experiment_1
'''

# GPU 설정 
global device
device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')

# TensorBoard 설정
global writer 
def file_set(path):
    global writer
    writer = SummaryWriter(path)

# TensorBoard에 학습 이미지 기록 
def write(net, train_loader, desc):
    # 이미지 가져오기
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images=images.to(device)

    # 이미지 그리드를 만들기
    img_grid = torchvision.utils.make_grid(images)

    # TensorBoard 기록 
    writer.add_image(desc, img_grid)
    writer.add_graph(net, images)
    writer.close()

# 학습 중 손실(running loss)을 기록
def loss_write(loss,step):
    writer.add_scalar('training loss',loss,step)
    writer.close()


# 헬퍼 함수
def add_pr_curve_tensorboard(classes, class_index, test_probs, test_label, global_step=0):
    '''
    0부터 9까지의 "class_index"를 가져온 후 해당 정밀도-재현율(precision-recall)
    곡선을 그립니다
    '''
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# TensorBoard로 학습된 모델 평가 
def eval_write(net,test_loader,classes):
    class_probs = []
    class_label = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images=images.to(device)
            labels=labels.to(device)
            output = net(images)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]

            class_probs.append(class_probs_batch)
            class_label.append(labels)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_label = torch.cat(class_label)

    # 모든 정밀도-재현율(precision-recall; pr) 곡선을 그림
    for i in range(len(classes)):
        add_pr_curve_tensorboard(classes,i, test_probs,test_label)