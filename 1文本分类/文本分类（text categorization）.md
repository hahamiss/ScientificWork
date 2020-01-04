# 文本分类（text categorization）

## 代码

### 搜狗数据库

bptree做文本分类：

```matlab
d='D:\experimental data\text categorization\SogouC.reduced\processed';
d2='D:\experimental data\text categorization\SogouC.reduced\processed\bptree';
cd(d);
%10-fold Cross Validation将每个主题数据分为10份
for C=1:10
for numepochs=1:10
for A=1:9
for B=1:9
if A<B

ct1=A;
d1=[d,'\',num2str(ct1)];
cd(d1)
eval(['num_word4_' num2str(ct1) '=[];']);
for ct2=1:1990
eval(['load num_word4_' num2str(ct1) '_' num2str(ct2) '.mat']);
eval(['num_word4_' num2str(ct1) '_' num2str(ct2) '=sparse(num_word4_' num2str(ct1) '_' num2str(ct2) ');']);
eval(['num_word4_' num2str(ct1) '=[num_word4_' num2str(ct1) ',num_word4_' num2str(ct1) '_' num2str(ct2) '];']);
eval(['clear num_word4_' num2str(ct1) '_' num2str(ct2)]);
%disp(ct2)
end
ct1=B;
d1=[d,'\',num2str(ct1)];
cd(d1)
eval(['num_word4_' num2str(ct1) '=[];']);
for ct2=1:1990
eval(['load num_word4_' num2str(ct1) '_' num2str(ct2) '.mat']);
eval(['num_word4_' num2str(ct1) '_' num2str(ct2) '=sparse(num_word4_' num2str(ct1) '_' num2str(ct2) ');']);
eval(['num_word4_' num2str(ct1) '=[num_word4_' num2str(ct1) ',num_word4_' num2str(ct1) '_' num2str(ct2) '];']);
eval(['clear num_word4_' num2str(ct1) '_' num2str(ct2)]);
%disp(ct2)
end
eval(['data1=num_word4_' num2str(A) ';']);
eval(['data2=num_word4_' num2str(B) ';']);
test1=data1(:,(C-1)*199+1:C*199);
test2=data2(:,(C-1)*199+1:C*199);
train1=data1;  train1(:,(C-1)*199+1:C*199)=[];
train2=data2;  train2(:,(C-1)*199+1:C*199)=[];

tic
eval(['W' num2str(C) '_' num2str(numepochs) '_' num2str(A) '_' num2str(B) '=BPtree_train(100000,10,train1,train2,numepochs);']);
time1=toc;

cd(d2)
eval(['save W' num2str(C) '_' num2str(numepochs) '_' num2str(A) '_' num2str(B) ' W' num2str(C) '_' num2str(numepochs) '_' num2str(A) '_' num2str(B)]);
cd(d1)

x=[test1,test2];
x=full(x');
x(:,100001:end)=[];
y=ones(398,1);
y(200:end)=-1;
tic
eval(['W=W' num2str(C) '_' num2str(numepochs) '_' num2str(A) '_' num2str(B) ';']);
nn=nnsetup_nobias([100000 10000 1000 100 10 1]);
nn.W=W;
nn.momentum=0;
nn.sparsityTarget=0;
nn.output='tanh_opt';
nn=nnff_nobias(nn,x,y);
l=0;
ls=nn.a{6};
for i=1:199
if ls(i)<0
l=l+1;
end
end
for i=200:398
if ls(i)>0
l=l+1;
end
end
lv_test=(398-l)/398;
time2=toc;
fprintf('测试集两两分类(%d-%d-%d-%d)准确率：%d;训练时长：%d；测试时长：%d\n',C,numepochs,A,B,lv_test,time1,time2)

end
end
end
end
end
```

10-fold Cross Validation

```matlab
%数据处理，全部读入
d='D:\experimental data\text categorization\SogouC.reduced\processed';
cd(d);
for i=1:9
d1=[d,'\',num2str(i)];
cd(d1)
temp=[];
for j=1:1990
eval(['load num_word4_' num2str(i) '_' num2str(j) '.mat;']);
eval(['temp=[temp,sparse(num_word4_' num2str(i) '_' num2str(j) ')];']);
eval(['clear num_word4_' num2str(i) '_' num2str(j) ';']);
fprintf('读取词包%d-%d\n',j,i)
end
text_all{i}=temp;
end
%10-fold Cross Validation
d1=[d,'\10-fold Cross Validation'];
cd(d1)
for i=1:10
for j=1:9
eval(['test_' num2str(i) '{j}=text_all{j}(:,(j-1)*199+1:j*199);']);
temp=text_all{j};temp(:,(j-1)*199+1:j*199)=[];
eval(['train_' num2str(i) '{j}=temp;']);
end
eval(['save test_' num2str(i) ' test_' num2str(i) ';']);
eval(['save train_' num2str(i) ' train_' num2str(i) ';']);
end
```

朴素贝叶斯文本分类

```matlab
d='D:\experimental data\text categorization\SogouC.reduced\processed\10-fold Cross Validation';
cd(d)

%for C=1:10
C=9;
eval(['load train_' num2str(C) '.mat']);
eval(['load test_' num2str(C) '.mat']);
eval(['train=train_' num2str(C) ';test=test_' num2str(C) ';']);
eval(['clear train_' num2str(C) ';clear test_' num2str(C) ';']);

for i=1:9
temp=train{i};
l=sum(temp')';
Pa_y{i}=l/sum(l);
end

for i=1:9
temp=test{i};
for j=1:size(temp,2)
fprintf('j-i(9):%d-%d\n',j,i)
x=temp(:,j);
for k=1:numel(x)
if x(k,1)>0
x(k,1)=1;
end
end
for m=1:9
Px_y(m,j)=full(sum(x.*Pa_y{m}));  %以加代乘
end
end
test_Py_x{i}=Px_y;
end

%计算9分类准确率
for i=1:9
temp=full(test_Py_x{i});
counter=0;
for j=1:size(temp,2)
ls=temp(:,j);
site=find(ls==max(ls));
if site==i
counter=counter+1;
end
end
lv(i)=counter/size(temp,2);
end
```























