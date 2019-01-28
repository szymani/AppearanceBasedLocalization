import matplotlib.pyplot as plt

# get the figure
f = plt.figure()

# x axis
x = [1, 2, 4, 8, 10, 16, 32]
xBar = [1, 2, 3, 4, 5]
label = ['64x64x64x2048', '48x48x48x4096', '64x64x64x4096', '80x80x80x4096', '96x96x96x4096']

# y axis
SGD = [0.5469, 0.5226, 0.5473, 0.7438, 0.7511, 0.8663, 0.3402]
Adam = [0.3214, 0.3367, 0.4682, 0.4977, 0.6374, 0.6592, 0.2875]
Struc = [0.5242, 0.7511, 0.7902, 0.8026, 0.8253]
# use LaTeX fonts in the plot
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# plot
plt.grid(True)
#plt.plot(x, SGD, '.:')
#plt.plot(x, Adam, '.:')
plt.bar(xBar, Struc, width = 0.6)
plt.xticks(xBar, label, rotation='-60')
#plt.legend(('Modified SGD', 'Adam'), loc='upper right')

# set labels (LaTeX can be used)
#plt.title(r'\textbf{Mutual Information Feature Selection}', fontsize=11)
#plt.xlabel(r'\textbf{Batch size}', fontsize=11)
#plt.ylabel(r'\textbf{Accuracy after 5000 iterations}', fontsize=11)

plt.ylabel(r'\textbf{Accuracy}', fontsize=11)


# save as PDF
f.savefig("structPlot.pdf", bbox_inches='tight')