@echo off
chcp 65001 > nul
echo ========================================================
echo       ASVspoof: Оценка моделей на пользовательском датасете
echo ========================================================
echo.

:: Проверка виртуального окружения
if not exist venv\Scripts\activate.bat (
    echo [ERROR] Виртуальное окружение 'venv' не найдено!
    echo Пожалуйста, сначала запустите 'install_deps.bat'.
    pause
    exit /b 1
)

call venv\Scripts\activate

:: Запрос имени папки датасета
set DATASET_NAME=%1
if "%DATASET_NAME%"=="" (
    echo Введите имя подпапки внутри директории 'data', где находится ваш датасет.
    echo (Например, если ваш датасет должен быть в data\custom_eval, введите custom_eval)
    echo.
    set /p DATASET_NAME="Имя подпапки: "
)

if "%DATASET_NAME%"=="" (
    echo [ERROR] Имя подпапки не может быть пустым.
    pause
    exit /b 1
)

set DATA_PATH=data\%DATASET_NAME%

:: Создаем папку, если она не существует
if not exist %DATA_PATH% (
    echo [INFO] Создание подпапки %DATA_PATH%...
    mkdir %DATA_PATH%
    echo [INFO] Подпапка %DATA_PATH% успешно создана.
    echo [INFO] Пожалуйста, поместите ваш архив (zip или tar.gz) или распакованные файлы в эту папку.
    echo.
    echo После того как вы скопируете файлы, нажмите любую клавишу для продолжения...
    pause
)

:check_unpack
python scripts/unpack_custom.py --dir %DATA_PATH%
set UNPACK_STATUS=%errorlevel%

if %UNPACK_STATUS% equ 2 (
    echo.
    echo [WARNING] Не удалось найти архивы (.zip, .tar.gz) или распакованные аудиофайлы (.flac, .wav) в %DATA_PATH%
    echo Пожалуйста, убедитесь, что вы скопировали архив или аудиофайлы в указанную папку.
    echo.
    echo Нажмите любую клавишу для повторной проверки...
    pause
    goto check_unpack
)

if %UNPACK_STATUS% neq 0 (
    echo [ERROR] Произошла ошибка при распаковке архива.
    pause
    exit /b 1
)

echo.
echo [INFO] Запуск оценки моделей на датасете %DATA_PATH%...
python src/evaluate_custom.py --data-dir %DATA_PATH%

if %errorlevel% neq 0 (
    echo [ERROR] Оценка завершилась с ошибкой.
) else (
    echo.
    echo ========================================================
    echo [SUCCESS] Оценка успешно завершена!
    echo Результаты сохранены в:
    echo   - Текстовый отчет: %DATA_PATH%\custom_evaluation_results.txt
    echo   - Оценки для каждого аудиофайла (CSV): %DATA_PATH%\custom_evaluation_scores.csv
    echo ========================================================
)

pause
