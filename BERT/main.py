import torch
from torch import nn
from torch import optim

from bert_model import bert
from data_processing import data_processing

if __name__ == '__main__':
    process = data_processing()
    batch = process.preprocess()
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))
    model = bert(process.vocab_size)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(100):
        optimizer.zero_grad()
        inputs = [input_ids, segment_ids, masked_pos]
        logits_lm, logits_clsf = model(inputs)
        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
        loss_lm = (loss_lm.float()).mean()
        logits_clsf = criterion(logits_clsf, isNext)
        loss = loss_lm + logits_clsf
        if (epoch + 1) % 10 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
            loss.backward()
            optimizer.step()
