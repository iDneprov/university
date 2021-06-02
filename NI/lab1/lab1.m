set(0, 'DefaultTextInterpreter', 'latex');
% Задание 1
% Входные данные
x = [3 -3.8 -1.8 -1.1 -3.2 -4.8; ...
    2.4 0.2  0.4 -0.9 -2.5 4.2];
y = [0 1 1 1 1 0];

% Создаем сеть и инициализируем ее случайными значениями
network = newp([-5 5; -5 5], [0 1]);
network.inputweights{1}.initFcn = 'rands'; % веса
network.biases{1}.initFcn = 'rands'; % смещения
network = init(network);
display(network);

% Обучаем ее две эпохи по правилу Розенблатта
epoches = 2;
learningCoeff = 0.1;

for epoch = 1:epoches
    for i = 1:length(x) % цикл по всем образцам
        out = network(x(:, i)); % выход сети для i-го образца
        error = y(:, i) - out;
        network.IW{1} = network.IW{1} + learningCoeff * error * x(:, i)'; % транспонируем '
        network.b{1} = network.b{1} + learningCoeff * error;
        fprintf('epochs: %d; iterations: %d; error: %.6e\n', epoch, i, mae(y - network(x)));
    end
    fprintf('W =\n%s\nb =\n%s\n', mat2str(network.IW{1}), mat2str(network.b{1}));
end

% Результаты обучения
plotpv(x, y);
plotpc(network.IW{1}, network.b{1});
grid on;
xlabel('$x_1$');
ylabel('$x_2$');

%% Инициализируем сеть случайными значениями и обучаем ее встроенной функцией
network = newp([-5 5; -5 5], [0 1]);
network.inputweights{1}.initFcn = 'rands';
network.biases{1}.initFcn = 'rands';
network = init(network);
network.trainParam.epochs = 50;
network = train(network, x, y);

% Результаты обучения с классификацией случайных точек
randomDotsX = rands(3, [-5 5; -5 5])'; % транспонируем'
randomDotsY = network(randomDotsX);

fprintf('W =\n%s\nb =\n%s\n', mat2str(network.IW{1}), mat2str(network.b{1}));

plotpv([x randomDotsX], [y randomDotsY]);
plotpc(network.IW{1}, network.b{1});
grid on;
xlabel('$x_1$');
ylabel('$x_2$');

%% Задание 2
% Обучение на линейнонеразделимом множестве
x = [3 -3.8 -1.8 -1.1 -3.2 -4.8; ...
      2.4 0.2 0.4 -0.9 -2.5 4.2];
y = [0 1 1 1 0 0];

plotpv(x, y);

% Инициализируем сеть случайными весами
network = newp([-5 5; -5 5], [0 1]);
network.inputweights{1}.initFcn = 'rands';
network.biases{1}.initFcn = 'rands';
network = init(network);

% Обучаем ее
network.trainParam.epochs = 50;
network = train(network, x, y);

% Результаты обучения
plotpv(x, y);
plotpc(network.IW{1}, network.b{1});
grid on;
xlabel('$x_1$');
ylabel('$x_2$');

%% Задание 3
% Классификация по 4 классам
% Входные данные
x = [2 2.3 0.4 -1.9 -3.2 -0.4 4.1 -5;...
     -1.3 4.5 0.4 -4.3 -4.1 -5 1.4 -4.7];
y = [0 0 0 1 1 1 0 1;...
     1 0 0 0 0 1 1 0];

% Инициализируем сеть случайными весами
network = newp([-5 5; -5 5], [0 1; 0 1]);
network.inputweights{1}.initFcn = 'rands';
network.biases{1}.initFcn = 'rands';
network = init(network);

% Обучаем ее
network.trainParam.epochs = 50;
network = train(network, x, y);

% Результаты обучения с классификацией случайных точек
randomDotsX = rands(5, [-5 5; -5 5])'; %'
randomDotsY = network(randomDotsX);

plotpv([x randomDotsX], [y randomDotsY]);
plotpc(network.IW{1}, network.b{1});
grid on;
xlabel('$x_1$');
ylabel('$x_2$');
