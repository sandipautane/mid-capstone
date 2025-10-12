import argparse
import os
import ssl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.resnet import ResNet50
from dataloader import get_imagenet_loaders, get_imagenet_subset_loaders

# Fix SSL certificate issue for downloading CIFAR-100
ssl._create_default_https_context = ssl._create_unverified_context

def get_cifar100_loaders(batch_size=128, num_workers=2):
    """Load CIFAR-100 dataset with data augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def train(net, trainloader, criterion, optimizer, device, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = train_loss / (batch_idx + 1)
    accuracy = 100. * correct / total

    # Print training results
    print('Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
          % (avg_loss, accuracy, correct, total))
    
    return avg_loss, accuracy


def test(net, testloader, criterion, device, epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / (batch_idx + 1)
    accuracy = 100. * correct / total

    # Print validation results
    print('Val Loss: %.3f | Val Acc: %.3f%% (%d/%d)'
          % (avg_loss, accuracy, correct, total))

    return avg_loss, accuracy

def save_checkpoint(net, optimizer, epoch, best_acc, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, filepath)


def load_checkpoint(filepath, net, optimizer):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    return epoch, best_acc


def plot_results(epochs, train_losses, train_accuracies, val_losses, val_accuracies, save_path=None):
    """Plot training and validation results."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Train ResNet50 on ImageNet 1K')
    parser.add_argument('--epochs', type=int, default=90, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--resume', type=str, default='', help='path to latest checkpoint')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='directory to save checkpoints')
    parser.add_argument('--data-dir', type=str, default='./data/imagenet', help='path to ImageNet dataset')
    parser.add_argument('--subset', action='store_true', help='use subset of ImageNet for faster training')
    parser.add_argument('--subset-size', type=int, default=10000, help='size of subset to use')
    parser.add_argument('--num-workers', type=int, default=8, help='number of data loading workers')
    parser.add_argument('--plot', action='store_true', help='plot training results')
    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Load data
    if args.subset:
        print(f'Loading ImageNet subset ({args.subset_size} samples)...')
        trainloader, valloader = get_imagenet_subset_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            subset_size=args.subset_size
        )
    else:
        print('Loading full ImageNet dataset...')
        trainloader, valloader = get_imagenet_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

    # Initialize model for ImageNet (1000 classes, 224x224 input)
    net = ResNet50(num_classes=1000, input_size=224)
    net = net.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Use ImageNet-style learning rate schedule
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

    # Initialize training tracking
    start_epoch = 0
    best_acc = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f'Loading checkpoint {args.resume}')
            start_epoch, best_acc = load_checkpoint(args.resume, net, optimizer)
            print(f'Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%')
        else:
            print(f'No checkpoint found at {args.resume}')

    # Training loop
    print('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train(net, trainloader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validate
        val_loss, val_acc = test(net, valloader, criterion, device, epoch)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Update learning rate
        scheduler.step()

        # Save checkpoint if best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(net, optimizer, epoch, best_acc, 
                          os.path.join(args.save_dir, 'best_model.pth'))
            print(f'New best accuracy: {best_acc:.2f}%')

        # Save regular checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(net, optimizer, epoch, best_acc, 
                          os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth'))

    print(f'Training completed. Best accuracy: {best_acc:.2f}%')

    # Plot results
    if args.plot:
        epochs = range(len(train_losses))
        plot_results(epochs, train_losses, train_accuracies, val_losses, val_accuracies,
                    os.path.join(args.save_dir, 'training_plots.png'))


if __name__ == '__main__':
    main()