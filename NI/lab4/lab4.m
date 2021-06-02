set(0, 'DefaultTextInterpreter', 'latex');

% Этап 1
% Задаем три элипса
t = 0:0.025:2*pi;

a = 0.2;
b = 0.2;
alpha = 0;
x0 = 0.2;
y0 = 0;
firstCurve = [cos(alpha), -sin(alpha); sin(alpha),  cos(alpha)] * [a * cos(t); b * sin(t)] + [x0 * ones(1, length(t)); y0 * ones(1, length(t))];

a = 0.7;
b = 0.5;
alpha = -pi/3;
x0 = 0;
y0 = 0;
secondCurve = [cos(alpha), -sin(alpha); sin(alpha),  cos(alpha)] * [a * cos(t); b * sin(t)] + [x0 * ones(1, length(t)); y0 * ones(1, length(t))];

a = 1;
b = 1;
alpha = 0;
x0 = 0;
y0 = 0;
thirdCurve = [cos(alpha), -sin(alpha); sin(alpha),  cos(alpha)] * [a * cos(t); b * sin(t)] + [x0 * ones(1, length(t)); y0 * ones(1, length(t))];

% Показываем эти элипсы
plot(firstCurve(1, :), firstCurve(2, :), 'r', secondCurve(1, :), secondCurve(2, :), 'g', thirdCurve(1, :), thirdCurve(2, :), 'b');
legend('First', 'Second', 'Third');
axis([-1.2 1.2 -1.2 1.2]);
grid on;

%% Создаем множество точек и разделяем его
% Генерируем точки, принадлежашие элипсам
firstDots = firstCurve(:, randperm(end, 60));
secondDots = secondCurve(:, randperm(end, 100));
thirdDots = thirdCurve(:, randperm(end, 120));

% Делим точки
[firstTraining, firstControl, firstTest] = dividerand(firstDots, 0.8, 0.0, 0.2);
[secondTraining, secondControl, secondTest] = dividerand(secondDots, 0.8, 0.0, 0.2);
[thirdTraining, thirdControl, thirdTest] = dividerand(thirdDots, 0.8, 0.0, 0.2);

trainingSetSize = length(firstTraining) + length(secondTraining) + length(thirdTraining);
n_val = length(firstControl) + length(secondControl) + length(thirdControl);
testSetSize = length(firstTest) + length(secondTest) + length(thirdTest);

% Демонстрация

p = plot(firstCurve(1, :), firstCurve(2, :), '-r', firstTraining(1, :), firstTraining(2, :), 'or',  firstControl(1, :),   firstControl(2, :),   'rV', firstTest(1, :),  firstTest(2, :),  'rs', ...
         secondCurve(1, :), secondCurve(2, :), '-g', secondTraining(1, :), secondTraining(2, :), 'og',  secondControl(1, :),   secondControl(2, :),   'gV', secondTest(1, :),  secondTest(2, :),  'gs', ...
         thirdCurve(1, :), thirdCurve(2, :), '-b', thirdTraining(1, :), thirdTraining(2, :), 'ob',  thirdControl(1, :),   thirdControl(2, :),   'bV', thirdTest(1, :),   thirdTest(2, :), 'bs');

mSize = 5;
lWidth = 1;
edgeColor = 'black';
faceColor = 'white';

p(1).LineWidth = lWidth;
p(2).MarkerEdgeColor = edgeColor;
p(2).MarkerFaceColor = 'r';
p(2).MarkerSize = mSize;
p(3).MarkerEdgeColor = edgeColor;
p(3).MarkerFaceColor = faceColor;
p(3).MarkerSize = mSize;

p(4).LineWidth = lWidth;
p(5).MarkerEdgeColor = edgeColor;
p(5).MarkerFaceColor = 'g';
p(5).MarkerSize = mSize;
p(6).MarkerEdgeColor = edgeColor;
p(6).MarkerFaceColor = faceColor;
p(6).MarkerSize = mSize;

p(7).LineWidth = lWidth;
p(8).MarkerEdgeColor = edgeColor;
p(8).MarkerFaceColor = 'b';
p(8).MarkerSize = mSize;
p(9).MarkerEdgeColor = edgeColor;
p(9).MarkerFaceColor = faceColor;
p(9).MarkerSize = mSize;

axis([-1.2 1.2 -1.2 1.2]);
legend('first: Curve', 'training', 'test', ...
      'second: Curve', 'training', 'test', ...
      'third: Curve', 'training', 'test');
grid on;

%% Этапы 2 и 3
% Готовим данные для обучения
xTrainig = [firstTraining secondTraining thirdTraining];
yTrainig = [1 * ones(1, length(firstTraining)) 2 * ones(1, length(secondTraining)) 3 * ones(1, length(thirdTraining))];

xTest = [firstTest secondTest thirdTest];
yTest = [1 * ones(1, length(firstTest)) 2 * ones(1, length(secondTest)) 3 * ones(1, length(thirdTest))];

% Создаем и обучаем сеть
SPREAD = 0.3;
network = newpnn(xTrainig, ind2vec(yTrainig), SPREAD);

% Смотрим насколько хорошо сеть обучилась
correctTrainig = sum(vec2ind(sim(network, xTrainig)) == yTrainig);
correctTest = sum(vec2ind(sim(network, xTest)) == yTest);
fprintf('Обучающее множество: %d:%d\nТестовое множество: %d:%d\n', correctTrainig, trainingSetSize, correctTest, testSetSize);

% Разделение по классам на картине
h = 0.025;
n = int32((1.2 + 1.2) / h) + 1;
x = zeros(2, n * n);

for i = 1:n
    for j = 1:n
        x(:, (i-1)*n + j) = [-1.2 + (double(i)-1)*h; 1.2 - (double(j)-1)*h];
    end
end

image(permute(reshape(sim(network, x), [3 n n]), [2 3 1]));

%% Создаем и обучаем сеть
SPREAD = 0.1;
network = newpnn(xTrainig, ind2vec(yTrainig), SPREAD);

% Смотрим насколько хорошо сеть обучилась
correctTrainig = sum(vec2ind(sim(network, xTrainig)) == yTrainig);
correctTest = sum(vec2ind(sim(network, xTest)) == yTest);
fprintf('Обучающее множество: %d:%d\nТестовое множество: %d:%d\n', correctTrainig, trainingSetSize, correctTest, testSetSize);

% Разделение по классам на картине
h = 0.025;
n = int32((1.2 + 1.2) / h) + 1;
x = zeros(2, n * n);

for i = 1:n
    for j = 1:n
        x(:, (i-1)*n + j) = [-1.2 + (double(i)-1)*h; 1.2 - (double(j)-1)*h];
    end
end

image(permute(reshape(sim(network, x), [3 n n]), [2 3 1]));

%% Создаем и обучаем сеть
SPREAD = 0.3;
network = newrb(xTrainig, ind2vec(yTrainig), 1e-5, SPREAD);

% Смотрим насколько хорошо сеть обучилась
correctTrainig = sum(vec2ind(sim(network, xTrainig)) == yTrainig);
correctTest = sum(vec2ind(sim(network, xTest)) == yTest);
fprintf('Обучающее множество: %d:%d\nТестовое множество: %d:%d\n', correctTrainig, trainingSetSize, correctTest, testSetSize);

% Разделение по классам на картине
h = 0.025;
n = int32((1.2 + 1.2) / h) + 1;
x = zeros(2, n * n);

for i = 1:n
    for j = 1:n
        x(:, (i-1)*n + j) = [-1.2 + (double(i)-1)*h; 1.2 - (double(j)-1)*h];
    end
end

image(permute(reshape(sim(network, x), [3 n n]), [2 3 1]));

%% Создаем и обучаем сеть
SPREAD = 0.1;
network = newrb(xTrainig, ind2vec(yTrainig), 1e-5, SPREAD);

%% Смотрим насколько хорошо сеть обучилась
correctTrainig = sum(vec2ind(sim(network, xTrainig)) == yTrainig);
correctTest = sum(vec2ind(sim(network, xTest)) == yTest);
fprintf('Обучающее множество: %d:%d\nТестовое множество: %d:%d\n', correctTrainig, trainingSetSize, correctTest, testSetSize);

% Разделение по классам на картине
h = 0.025;
n = int32((1.2 + 1.2) / h) + 1;
x = zeros(2, n * n);

for i = 1:n
    for j = 1:n
        x(:, (i-1)*n + j) = [-1.2 + (double(i)-1)*h; 1.2 - (double(j)-1)*h];
    end
end

image(permute(reshape(sim(network, x), [3 n n]), [2 3 1]));

%% Этап 2 и 3
% Аппроксимация функции
f = @(t) sin(t .^ 2 - 7 * t);
t = 0:0.025:5;

X = t;
y = f(t);

% Оставляем с конца 10%
trainingSetSize = ceil(length(X) * 0.9);
xTrainig = X(1:trainingSetSize);
yTrainig = y(1:trainingSetSize);
xTest = X(trainingSetSize+1:end);
yTest = y(trainingSetSize+1:end);

% Создаем и обучаем нейросеть
network = newgrnn(xTrainig, yTrainig, 0.01);

%% Результаты обучения на обучающем подмножестве
disp(accuracy(sim(network, xTrainig), yTrainig));
p = plot(xTrainig, yTrainig, xTrainig, sim(network, xTrainig), 'o');
p(1).Color = [1 0 0];
p(2).MarkerSize = 3;
p(2).Color = [0 0 0];
xlabel('$t$');
ylabel('$y$');
grid on;

%% Результаты обучения на тестовом подмножестве
disp(accuracy(sim(network, xTest), yTest));
p = plot(xTest, yTest, xTest, sim(network, xTest), 'o');
p(1).Color = [1 0 0];
p(2).MarkerSize = 3;
p(2).Color = [0 0 0];
xlabel('$t$');
ylabel('$y$');
grid on;

%% Делим подмножества в заданном соотношении
[tariningSplit, x, testSplit] = dividerand(1:length(X), 0.8, 0.0, 0.2);
trainingSetSize = length(tariningSplit);
testSetSize = length(testSplit);
xTrainig = X(tariningSplit);
yTrainig = y(tariningSplit);
xTest = X(testSplit);
yTest = y(testSplit);
network = newgrnn(xTrainig, yTrainig, 0.01);

% Результаты обучения на обучающем подмножестве
disp(accuracy(sim(network, xTrainig), yTrainig));
p = plot(X, y, xTrainig, sim(network, xTrainig), 'o');
p(1).Color = [1 0 0];
p(2).MarkerSize = 3;
p(2).Color = [0 0 0];
xlabel('$t$');
ylabel('$y$');
grid on;

%% Результаты обучения на тестовом подмножестве
disp(accuracy(sim(network, xTest), yTest));
p = plot(X, y, xTest, sim(network, xTest), 'o');
p(1).Color = [1 0 0];
p(2).MarkerSize = 3;
p(2).Color = [0 0 0];
xlabel('$t$');
ylabel('$y$');
grid on;
