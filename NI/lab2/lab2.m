set(0, 'DefaultTextInterpreter', 'latex');
% Задание 1
% Входные данные
signalFunction = @(t) sin(-2 .* t .^ 2 + 7 .* t) - 0.5 .* sin(t);
t = 0:0.025:4.5;
signal = signalFunction(t);
D = 1:5; % задержки
x = con2seq(signal(D(end)+1:end)); % образцы
y = con2seq(signal(D(end)+1:end)); % цели

% Создаем сеть и инициализируем ее случайными значениями
network = newlin([-1 1], 1, D, 0.01);
network.inputweights{1}.initFcn = 'rands';
network.biases{1}.initFcn = 'rands';
network = init(network);

% Обучаем ее с помощью adapt

for i = 1:50
    [network, ~, err, ~] = adapt(network, x, y, con2seq(signal(D)));
    fprintf('|iteration: %d | error: %f|\n', i, sqrt(mse(err)));
end

%% Результаты обучения
predictedSignal = cell2mat(network([con2seq(signal(D)) x]));
p = plot(t, signal, t(D(end)+1:end), predictedSignal(D(end)+1:end), 'o');
p(1).Color = [1 0 0];
p(2).MarkerSize = 3;
p(2).Color = [0 0 0];

grid on

xlabel('$t$');
ylabel('$y$');
legend('y', 'y_{predicted}');
%% Задание 2
% Входные данные
signalFunction = @(t) sin(t .^ 2 - 6 .* t + 3);
t = 0:0.025:6;
signal = signalFunction(t);
D = 1:3;  % задержки
x = con2seq(signal(D(end)+1:end)); % образцы
y = con2seq(signal(D(end)+1:end)); % цели

% Создаем сеть и инициализируем ее случайными значениями
network = newlin([-1 1], 1, D, maxlinlr(cell2mat(x), 'bias'));
network.inputweights{1}.initFcn = 'rands';
network.biases{1}.initFcn = 'rands';
network = init(network);

% Обучаем ее с помощью train
network.trainParam.epochs = 600;
network.trainParam.goal = 1e-6;
network = train(network, x, y, con2seq(signal(D)));

%% Результаты обучения
predictedSignal = cell2mat(network([con2seq(signal(D)) x]));
p = plot(t, signal,  t(D(end)+1:end), predictedSignal(D(end)+1:end), 'o');
p(1).Color = [1 0 0];
p(2).MarkerSize = 3;
p(2).Color = [0 0 0];
grid on
xlabel('$t$');
ylabel('$y$');
legend('y', 'y_{predicted}');

% Данные для таблицы
display(dataForTable(cell2mat(y), predictedSignal(D(end)+1:end)));

%% Вычисляем прогноз
t = 4.7:0.025:5;
signal = signalFunction(t);  %% len = 13
predictedSignal = [signal(1:3) zeros(1, 10)];
for i = 4:13
    tmp = cell2mat(network(con2seq(predictedSignal(i-3:i))));
    predictedSignal(i) = tmp(end);
end

% Строим график
p = plot(t, signal, t(4:end), predictedSignal(4:end), 'o');
p(1).Color = [1 0 0];
p(2).MarkerSize = 3;
p(2).Color = [0 0 0];
grid on
xlabel('$t$');
ylabel('$y$');
legend('y', 'y_{predicted}');

% Данные для таблицы
display(dataForTable(signal(4:end), predictedSignal(4:end)));

%% Задание 3
% Входные данные
t = 0:0.01:6;
x = sin(t .^ 2 - 6 * t + 3);
y = 1 / 3 * sin(t .^ 2 - 6 * t + pi / 6);
%x = cos(-5 * t .^ 2 + 10 * t - 5);            % входной сигнал
%y = 1 / 8 * cos(-5 * t .^ 2 + 10 * t);          % выходной сигнал

D = 4;         % глубина
Q = length(x); % число образцов

P = zeros(D, length(x));
for i = 1:D
    P(i,i:Q) = x(1:Q-i+1);
end

%% Создание сети + график + характеристики

network = newlind(P, y);

fprintf('W = %s\nb = %s\n\n', mat2str(network.IW{1}), mat2str(network.b{1}));

display(dataForTable(y, network(P)));

p = plot(t, y, t, network(P), 'o');

p(1).Color = [1 0 0];
p(2).MarkerSize = 3;
p(2).Color = [0 0 0];

grid on
xlabel('$t$');
ylabel('$y$');
legend('y', 'y_{pred}');
