% Задание 1
% Выгружаем данные из файла
data = load('100years.txt');
nums = data(:, 4);
% Сглаживание траектории усредяющим фильтром с шириной окна 12
x = smooth(nums, 12);
% Получаем матрицу строк
x = con2seq(x(1:650)');

% Создаем сеть

D = 5;
network = timedelaynet(1:D, 8, 'trainlm');
network.layers{1}.transferFcn = 'tansig';
network.layers{2}.transferFcn = 'purelin';

% Разбиваем данные на подмножества
network.divideFcn = 'divideind';
network.divideParam.trainingIndexes = 1 : 500;
network.divideParam.validationIndexes = 501 : 600;
network.divideParam.testIndexes = 601 : 650;

% Конфигурируем и инициализируем нейросеть
network = configure(network, x, x);
network = init(network);
view(network);

%% Обучаем ее
network.trainParam.epochs = 600;
network.trainParam.max_fail = 600;
network.trainParam.goal = 10e-5;

%%Подготовка входных и целевых временных серий данных для моделирования или обучения
% Xs Сдвинутые входы
% Xi Начальные состояния задержки ввода
% Ai Состояния задержки начального слоя
% Ts Сдвинутые цели
[Xs, Xi, Ai, Ts] = preparets(network, x, x);

% Получаем значения предсказанные сетью
network = train(network, Xs, Ts, Xi, Ai);
Y = sim(network, Xs, Xi);

%% Строим графики
figure;
hold on;
grid on;
plot(cell2mat(x), '-b');
plot([cell2mat(Xi), cell2mat(Y)], '-r');

%%
figure;
hold on;
grid on;
error = cell2mat(x)- [cell2mat(Xi), cell2mat(Y)];
plot(error, '-g');

xCheck = cell2mat(x);
yCheck = cell2mat(Y);

%%
figure;
hold on;
grid on;
plot(xCheck(601 : 650), '-b');
plot(yCheck(601 - D : 650 - D), '-r');

%%
figure;
hold on;
grid on;
error = xCheck(601 : 650)- yCheck(601 - D : 650 - D);
plot(error, '-g');

%% Задание 2
% Генерируем обучающее множесво
k = 0 : 0.025 : 1;
p1 = sin(4 * pi * k);
t1 = -ones(size(k));

k = 0.46 : 0.025 : 3.01;
g = @(k) cos( -3 * k .^ 2 + 5 * k + 10) + 0.8;
p2 = g(k);
t2 = ones(size(k));

R = {0; 2; 2};
P = [repmat(p1, 1, R{1}), p2, repmat(p1, 1, R{2}), p2, repmat(p1, 1, R{3}), p2];
T = [repmat(t1, 1, R{1}), t2, repmat(t1, 1, R{2}), t2, repmat(t1, 1, R{3}), t2];

PPrepared = con2seq(P);
TPrepared = con2seq(T);

% Создаем сеть
network = distdelaynet({0 : 4, 0 : 4}, 8, 'trainoss');
network.layers{1}.transferFcn = 'tansig';
network.layers{2}.transferFcn = 'tansig';
network.divideFcn = '';
network = configure(network, PPrepared, TPrepared);
view(network);

%% Обучаем ее
network.trainParam.epochs = 100;
network.trainParam.goal = 10e-5;
[Xs, Xi, Ai, Ts] = preparets(network, PPrepared, TPrepared);
network = train(network, Xs, Ts, Xi, Ai);

%% Рассчитываем выход сети и отображаем результаты на графике
Y = sim(network, Xs, Xi);
figure;
hold on;
grid on;
plot(cell2mat(TPrepared), '-b');
plot([cell2mat(Xi) cell2mat(Y)], '-r');
legend('Source', 'Predicted');

%% Преобразовываем значения
Yc = zeros(1, numel(Xi) + numel(Y));
for i = 1 : numel(Xi)
    if Xi{i} >= 0
        Yc(i) = 1;
    else
        Yc(i) = -1;
    end
end
for i = numel(Xi) + 1 : numel(Y)
    if Y{i} >= 0
        Yc(i) = 1;
    else
        Yc(i) = -1;
    end
end

% Сравнивам полученные значения с эталонными
display( nnz(Yc == cell2mat(TPrepared)) / length(TPrepared) )


R = {8; 2; 2};
P = [repmat(p1, 1, R{1}), p2, repmat(p1, 1, R{2}), p2, repmat(p1, 1, R{3}), p2];
T = [repmat(t1, 1, R{1}), t2, repmat(t1, 1, R{2}), t2, repmat(t1, 1, R{3}), t2];

PPrepared = con2seq(P);
TPrepared = con2seq(T);

[Xs, Xi, Ai, Ts] = preparets(network, PPrepared, TPrepared);

Y = sim(network, Xs, Xi);

figure;
hold on;
grid on;
plot(cell2mat(TPrepared), '-b');
plot([cell2mat(Xi) cell2mat(Y)], '-r');
legend('Source', 'Predicted');

Yc = zeros(1, numel(Xi) + numel(Y));
for i = 1 : numel(Xi)
    if Xi{i} >= 0
        Yc(i) = 1;
    else
        Yc(i) = -1;
    end
end
for i = numel(Xi) + 1 : numel(Y)
    if Y{i} >= 0
        Yc(i) = 1;
    else
        Yc(i) = -1;
    end
end

display( nnz(Yc == cell2mat(TPrepared)) / length(TPrepared) )

%% Задание 3
% Задаем функции
uFunction = @(k) sin(k.^2 - 15 .* k + 3) - sin(k);
yNextFunction = @(y, u) y ./ (1 + y.^2) + u.^3;

% Инициализируем константы
tBegin = 0;
tEnd = 10;
h = 0.01;
n = 1 + (tEnd - tBegin) / h;
u = zeros(1, n);

% Заполняем массивы
y = zeros(1, n);
u(1) = uFunction(0);
for i = 2 : n
    t = tBegin + (i - 1) * h;
    y(i) = yNextFunction(y(i - 1), u(i - 1));
    u(i) = uFunction(t);
end

% Задаем глубину погружения и выделяем подмножества
x = u;
D = 3;
trainingIndexes = 1 : 700;
validationIndexes = 701 : 900;
testIndexes = 901 : 997;

% Создаем сеть
network = narxnet(1 : 3, 1 : 3, 10);
network.trainFcn = 'trainlm';
network.layers{1}.transferFcn = 'tansig';
network.layers{2}.transferFcn = 'purelin';
network.divideFcn = 'divideind';
network.divideParam.trainingIndexes = trainingIndexes;
network.divideParam.validationIndexes = validationIndexes;
network.divideParam.testIndexes = testIndexes;
network = init(network);
view(network);

%% Обучаем ее
network.trainParam.epochs = 600;
network.trainParam.max_fail = 600;
network.trainParam.goal = 1.0e-8;
[Xs, Xi, Ai, Ts] = preparets(network, con2seq(x), {}, con2seq(y));
network = train(network, Xs, Ts, Xi, Ai);

% Вычисляем выход сети
Y = sim(network, Xs, Xi, Ai);

%% Строим графики
figure;
hold on;
grid on;
plot(tBegin : h : tEnd, x, '-b', tBegin : h : tEnd, [x(1:D) cell2mat(Y)], '-r')
legend('Source', 'Predicted');

%%
figure;
hold on;
grid on;
plot(tBegin+D*h : h : tEnd, x(D+1:end) - cell2mat(Y), '-r')
legend('Train Error');

%%
xValid = x( validationIndexes( length(validationIndexes) - (D - 1) : length(validationIndexes) ) );
uValid = u( validationIndexes( length(validationIndexes) - (D - 1) : length(validationIndexes) ) );
inpuT   = [con2seq( u(testIndexes)  ); con2seq( x(testIndexes) )];
delay = [con2seq( uValid ); con2seq(xValid)];
predictTest = sim(network, inpuT, delay, Ai);

figure;
hold on;
grid on;
plot(x(testIndexes), '-b')
plot(cell2mat(predictTest), '-r');
legend('Source', 'Predicted');

%%
figure;
hold on;
grid on;
error = cell2mat(predictTest) - x(testIndexes);
plot(error, '-r');
legend('Test Error');
