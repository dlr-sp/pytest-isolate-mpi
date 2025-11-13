from __future__ import annotations

import argparse
import sys


def assemble_sub_pytest_cmd(option: argparse.Namespace, nodeid: str):
    cmd = [_get_python_executable(option), "-m", "mpi4py", "-m", "pytest"]
    _add_general_options(cmd, option)
    _add_reporting_options(cmd, option)
    _add_pytest_warning_options(cmd, option)
    _add_collection_options(cmd, option)
    _add_test_session_options(cmd, option)
    _add_logging_options(cmd, option)
    # TODO: Coverage options
    cmd += [nodeid]  # test selection
    return cmd


def _get_python_executable(option):
    if option.python_executable:
        executable = option.python_executable
    else:
        executable = sys.executable
    return executable


def _add_general_options(cmd, option):
    cmd += ["--capture", option.capture]
    if option.runxfail:
        cmd += ["--runxfail"]


def _add_reporting_options(cmd, option):
    cmd += ["--no-header", "--no-summary"]
    if option.verbose:
        cmd += [f"-{option.verbose * 'v'}"]
    if option.disable_warnings:
        cmd += ["--disable-warnings"]
    if option.showlocals:
        cmd += ["--showlocals"]
    cmd += ["--tb", option.tbstyle]
    cmd += ["--show-capture", option.showcapture]
    if option.fulltrace:
        cmd += ["--full-trace"]
    cmd += ["--color", option.color]
    cmd += ["--code-highlight", option.code_highlight]


def _add_pytest_warning_options(cmd, option):
    if option.pythonwarnings is not None:
        for warning in option.pythonwarnings:
            cmd += ["--pythonwarnings", warning]
    if option.custom_ini_for_subsession:
        cmd += ["--config-file", option.custom_ini_for_subsession]
    elif option.inifilename is not None:
        cmd += ["--config-file", option.inifilename]


def _add_collection_options(cmd, option):
    if option.confcutdir is not None:
        cmd += ["--confcutdir", option.confcutdir]
    if option.noconftest:
        cmd += ["--noconftest"]
    cmd += ["--import-mode", option.importmode]


def _add_test_session_options(cmd, option):
    for plugin in option.plugins:
        cmd += ["-p", plugin]
    if option.traceconfig:
        cmd += ["--trace-config"]
    if option.override_ini is not None:
        for override in option.override_ini:
            cmd += ["--override-ini", override]
    if option.assertmode != "rewrite":
        cmd += ["--assert", option.assertmode]
    if option.setuponly:
        cmd += ["--setup-only"]
    if option.setupshow:
        cmd += ["--setup-show"]
    if option.setupplan:
        cmd += ["--setup-plan"]


def _add_logging_options(cmd, option):
    if option.log_level is not None:
        cmd += ["--log-level", option.log_level]
    if option.log_format is not None:
        cmd += ["--log-format", option.log_format]
    if option.log_cli_level is not None:
        cmd += ["--log-cli-level", option.log_cli_level]
    if option.log_cli_format is not None:
        cmd += ["--log-cli-format", option.log_cli_format]
    if option.log_cli_date_format is not None:
        cmd += ["--log-cli-date-format", option.log_cli_date_format]
    if option.log_file is not None:
        cmd += ["--log-file", option.log_file]
    if option.log_file_level is not None:
        cmd += ["--log-file-level", option.log_file_level]
    if option.log_file_format is not None:
        cmd += ["--log-file-format", option.log_file_format]
    if option.log_file_date_format is not None:
        cmd += ["--log-file-date-format", option.log_file_date_format]
    if option.log_auto_indent is not None:
        cmd += ["--log-auto-indent", str(option.log_auto_indent)]
    for disable in option.logger_disable:
        cmd += ["--log-disable", disable]
