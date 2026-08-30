#!/usr/bin/env ruby
# frozen_string_literal: true

project_path = ARGV.fetch(0, 'app/ios/Runner.xcodeproj/project.pbxproj')
profile_name = ARGV.fetch(1)

project = File.read(project_path)
release_configuration = /(
  \t\t97C147071CF9000F007C117D\s\/\*\sRelease\s\*\/\s=\s\{\n
  .*?
  \t\t\tbuildSettings\s=\s\{\n
)(.*?)(
  \t\t\t\};\n
  \t\t\tname\s=\sRelease;\n
  \t\t\};
)/mx

matches = project.scan(release_configuration)
abort 'Expected exactly one Runner Release build configuration' unless matches.length == 1

settings = matches.first[1].dup
automatic = "\t\t\t\tCODE_SIGN_STYLE = Automatic;\n"
abort 'Runner Release is no longer configured for automatic signing' unless settings.include?(automatic)

settings.sub!(automatic, <<~SETTINGS.gsub(/^/, "\t\t\t\t"))
  CODE_SIGN_IDENTITY = "Apple Distribution";
  CODE_SIGN_STYLE = Manual;
  PROVISIONING_PROFILE_SPECIFIER = "#{profile_name}";
SETTINGS

updated = project.sub(release_configuration) do
  "#{Regexp.last_match(1)}#{settings}#{Regexp.last_match(3)}"
end
abort 'Runner Release signing configuration was not updated' if updated == project

File.write(project_path, updated)
puts "Configured Runner Release for profile #{profile_name}"
