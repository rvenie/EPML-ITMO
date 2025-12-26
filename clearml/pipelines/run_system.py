#!/usr/bin/env python3
"""
Система быстрого запуска ClearML пайплайнов
Простое управление ClearML сервером и пайплайнами
"""

import logging
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

import requests

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClearMLSystemManager:
    """Менеджер системы ClearML."""

    def __init__(self):
        """Инициализация менеджера."""
        self.base_path = Path(__file__).parent.parent
        self.servers = {
            "web": "http://localhost:8080",
            "api": "http://localhost:8008",
            "files": "http://localhost:8081",
        }

    def check_servers_status(self) -> bool:
        """Проверяет доступность ClearML серверов."""
        logger.info("Проверка ClearML серверов...")

        all_ok = True
        for name, url in self.servers.items():
            try:
                if "8080" in url:
                    response = requests.get(url, timeout=5)
                else:
                    response = requests.get(f"{url}/debug.ping", timeout=5)
                # 200 = OK, 401 = requires auth (server is running)
                if response.status_code in (200, 401):
                    status_note = (
                        " (требует авторизацию)" if response.status_code == 401 else ""
                    )
                    logger.info(f"✅ {name} сервер: {url}{status_note}")
                else:
                    logger.warning(
                        f"⚠️ {name} сервер не отвечает: {url} (статус: {response.status_code})"
                    )
                    all_ok = False
            except Exception as e:
                logger.error(f"❌ {name} сервер недоступен: {e}")
                all_ok = False

        return all_ok

    def start_clearml_server(self) -> bool:
        """Запускает ClearML сервер через Docker Compose."""
        logger.info("Запуск ClearML сервера...")

        docker_compose_file = self.base_path / "config" / "docker-compose-clearml.yml"

        if not docker_compose_file.exists():
            logger.error(f"Docker Compose файл не найден: {docker_compose_file}")
            return False

        try:
            cmd = ["docker-compose", "-f", str(docker_compose_file), "up", "-d"]
            result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603

            if result.returncode == 0:
                logger.info("ClearML сервер запускается...")
                logger.info("Ожидание инициализации (60 секунд)...")

                # Показываем прогресс
                for i in range(6):
                    print(f"Ожидание: {60 - i * 10} секунд...", end="\r")
                    time.sleep(10)
                print()

                return True
            else:
                logger.error(f"Ошибка запуска: {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"Ошибка запуска ClearML сервера: {e}")
            return False

    def run_test_pipeline(self) -> bool:
        """Запускает тестовый пайплайн."""
        logger.info("Запуск тестового ML пайплайна...")

        try:
            cmd = [sys.executable, "pipeline_scheduler.py", "test"]
            result = subprocess.run(  # nosec B603
                cmd,
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=300,  # 5 минут
            )

            if result.returncode == 0:
                logger.info("✅ Тестовый пайплайн выполнен")
                logger.info("Проверьте результаты: http://localhost:8080")
                return True
            else:
                logger.error(f"❌ Ошибка пайплайна: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("❌ Превышено время выполнения (5 минут)")
            return False
        except Exception as e:
            logger.error(f"❌ Ошибка запуска: {e}")
            return False

    def show_status(self):
        """Показывает статус системы."""
        print("\n" + "=" * 50)
        print("СТАТУС СИСТЕМЫ CLEARML")
        print("=" * 50)

        # Проверяем серверы
        servers_ok = self.check_servers_status()

        # Проверяем файлы
        required_files = [
            "ml_pipeline.py",
            "pipeline_scheduler.py",
            "pipeline_monitor.py",
        ]

        files_ok = True
        print("\n📁 ФАЙЛЫ СИСТЕМЫ:")
        for file_name in required_files:
            file_path = Path(__file__).parent / file_name
            if file_path.exists():
                print(f"✅ {file_name}")
            else:
                print(f"❌ {file_name}")
                files_ok = False

        print("\n" + "=" * 50)
        print("ОБЩИЙ СТАТУС:")
        print(f"Серверы: {'✅ ОК' if servers_ok else '❌ ПРОБЛЕМЫ'}")
        print(f"Файлы: {'✅ ОК' if files_ok else '❌ ПРОБЛЕМЫ'}")

        if servers_ok and files_ok:
            print("\n🚀 Система готова!")
        else:
            print("\n⚠️ Требуется настройка")
        print("=" * 50)

    def open_web_interface(self):
        """Открывает веб интерфейс ClearML."""
        try:
            import webbrowser

            webbrowser.open("http://localhost:8080")
            logger.info("Веб интерфейс открыт в браузере")
        except Exception as e:
            logger.error(f"Не удалось открыть браузер: {e}")
            print("Откройте вручную: http://localhost:8080")

    def interactive_menu(self):
        """Простое интерактивное меню."""
        while True:
            print("\n" + "=" * 40)
            print("CLEARML СИСТЕМА УПРАВЛЕНИЯ")
            print("=" * 40)
            print("1. Проверить статус")
            print("2. Запустить ClearML сервер")
            print("3. Тестовый пайплайн")
            print("4. Открыть веб интерфейс")
            print("0. Выход")
            print("=" * 40)

            try:
                choice = input("Выберите (0-4): ").strip()

                if choice == "0":
                    print("Выход из системы")
                    break
                elif choice == "1":
                    self.show_status()
                elif choice == "2":
                    self.start_clearml_server()
                elif choice == "3":
                    if self.check_servers_status():
                        self.run_test_pipeline()
                    else:
                        print("❌ ClearML сервер недоступен")
                elif choice == "4":
                    self.open_web_interface()
                else:
                    print("❌ Неверный выбор")

            except KeyboardInterrupt:
                print("\nВыход по Ctrl+C")
                break


def main():
    """Главная функция."""
    print("🚀 ClearML System Manager v1.0")

    manager = ClearMLSystemManager()

    # Обработка аргументов командной строки
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "status":
            manager.show_status()
        elif command == "server":
            manager.start_clearml_server()
        elif command == "test":
            if manager.check_servers_status():
                manager.run_test_pipeline()
            else:
                print("❌ Сначала запустите ClearML сервер:")
                print("python run_system.py server")
        elif command == "web":
            manager.open_web_interface()
        else:
            print(f"❌ Неизвестная команда: {command}")
            print("Доступные команды: status, server, test, web")
    else:
        # Интерактивный режим
        manager.interactive_menu()


if __name__ == "__main__":
    main()
