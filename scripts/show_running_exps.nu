#!/usr/bin/env nu

ps --long | where command =~ "python -m embodiedbench" | each { |command|
    print "----------------------------------------"
    print $"PID: ($command.pid) | Started: ($command.start_time)"
    print ($command.command)
}
