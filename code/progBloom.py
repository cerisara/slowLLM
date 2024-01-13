from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def perplex(mod,tok):
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-v1")
    d = dataset['validation']
    curtxt = []
    for i,json in enumerate(d):
        utt = json['text']
        if len(utt)>5: curtxt.append(utt)
        # debug
        if len(curtxt)>50: break
    print("nutts",len(curtxt))

    chunk_size = 1024
    tokens = tok(curtxt)['input_ids']
    concat = sum(tokens, [])
    total_length = len(concat)
    total_length = (total_length // chunk_size) * chunk_size
    fichtoks = [concat[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    print("n1024utts",len(fichtoks))

    with torch.no_grad():
        totloss = 0.
        floss = torch.nn.CrossEntropyLoss()
        for i,xx in enumerate(fichtoks):
            data = torch.LongTensor(xx[:-1]).view(1,-1)
            target = torch.LongTensor(xx[1:]).view(1,-1)
            output = mod(data,labels=target)
            loss = output['loss']
            totloss += loss.item()
        print('Loss: {:.6f} {} {} Shape: {}'.format(totloss/float(i), totloss, i, data.shape))
        # Loss: 14.468861 43.406582832336426 3 Shape: torch.Size([1, 1023])

tok = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
mod = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
perplex(mod,tok)
exit()


utt = "the time has come to apologize to Nature"
prompt = tok(utt)
tokids = prompt['input_ids']
x = torch.LongTensor([tokids])
out = mod(x)
print(out)
