import numpy as np

def compress_mel(filter_banks):
    (num_mel_bins,num_spectral_bins) = filter_banks.shape
    runs = ''
    mels = ''
    mel = 0
    current_run = 0
    for i in range(num_spectral_bins):
        if filter_banks[mel, i] == 0.0:
            mel += 1
            runs += '%d, ' % (current_run)
            current_run = 0
        current_run += 1
        mels += '    (int)(%f * MEL_ONE_VALUE + 0.5),\n' % (filter_banks[mel, i])
    if current_run != 0:
        runs += '%d' % (current_run)
    
    return """
#define MEL_BINS %d
#define MEL_SPECTRAL_BINS %d
int mel_bins_in_overlap[MEL_BINS] = { %s };
int mel_coefficients[MEL_SPECTRAL_BINS] = {
%s};""" %(num_mel_bins,num_spectral_bins, runs, mels)

filter_banks = np.zeros((64, 257))
j = 0
for i in range(0, 16):
    filter_banks[i, j] = 1.0
    j += 1
for i in range(16, 32):
    for k in range(2):
        filter_banks[i, j] = 1.0 - k / 2
        filter_banks[i+1, j] = k / 2
        j += 1
for i in range(32, 48):
    for k in range(4):
        filter_banks[i, j] = 1.0 - k / 4
        filter_banks[i+1, j] = k / 4
        j += 1
for i in range(48, 63):
    for k in range(10):
        filter_banks[i, j] = 1.0 - k / 10
        filter_banks[i+1, j] = k / 10
        if j == 256:
            break
        j += 1
print(filter_banks)
print(compress_mel(filter_banks))
data = np.zeros((257))
for i in range(257):
        if (i & 1):
            data[i] = i*3 + 1000
        else:
            data[i] = -i
data2=np.inner(filter_banks, data)
print(data2)
print(np.inner(filter_banks.transpose(), data2))
