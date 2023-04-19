from torch.utils.data import Dataset


class NEPDataset(Dataset):
    def __init__(self, post_simcse, avg_mac, avg_mic, p_mac, p_mic, avg_mic_mic, token, label):
        super().__init__()
        self.post_simcse = post_simcse
        self.avg_mac = avg_mac
        self.avg_mic = avg_mic
        self.p_mac = p_mac
        self.p_mic = p_mic
        self.avg_mic_mic = avg_mic_mic
        self.token = token
        self.label = label

    def __getitem__(self, index):
        item = {
            'post_simcse': self.post_simcse[index],
            'avg_mac': self.avg_mac[index],
            'avg_mic': self.avg_mic[index],
            'kernel_p_mac': self.p_mac[index],
            'kernel_p_mic': self.p_mic[index],
            'kernel_avg_mic_mic': self.avg_mic_mic[index],
            'token': self.token[index],
            'label': self.label[index]
        }
        return item

    def __len__(self):
        return len(self.label)
