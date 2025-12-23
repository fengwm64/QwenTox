import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from TCC.model.LSTM import lstm
from TCC.model.BILSTM import bilstm
from tqdm import tqdm


def train(train_loader, val_loader, epochs, learning_rate, filename, vocab, PAD_IDX):
    if filename == "lstm":
        model = lstm(vocab_size=len(vocab), embed_dim=50, hidden_dim=256, output_dim=6, pad_idx=PAD_IDX)
    elif filename == "bilstm":
        model = bilstm(vocab_size=len(vocab), embed_dim=50, hidden_dim=256, output_dim=6, pad_idx=PAD_IDX)
    else:
        print("model load Error!")
        sys.exit(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for texts, labels, lengths in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            # Move data to device
            texts, labels = texts.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            # Calculate metrics
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).all(dim=1).sum().item()
            total += labels.size(0)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for texts, labels, lengths in val_loader:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts, lengths)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == labels).all(dim=1).sum().item()
                val_total += labels.size(0)

        # Epoch Statistics
        train_loss /= len(train_loader)
        train_acc = correct / total
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join("../../save", f"{filename}.pth")
            torch.save(model.state_dict(), save_path)

    print("Training complete!")