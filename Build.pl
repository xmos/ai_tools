#!/usr/bin/perl
use warnings;
use strict;

use Carp;
use SetupEnv;
use File::Copy;
use File::Path;
use XmosBuildLib;
use XmosArg;

my $MAKE_FLAGS;
my $CONFIG;
my $HOST;
my $DOMAIN;
my $BIN;

my %ALIASES =
  (
	  'all' => ['clean','build','install'],
  );

my %TARGETS =
  (
   'clean' => [\&DoClean, "Clean"],
   'build' => [\&DoBuild, "Build"],
   'install' => [\&DoInstall, "Install packaged binaries"],
  );

sub main
{
  my $xmosArg = XmosArg::new(\@ARGV);
  SetupEnv::SetupPaths();

  my @targets =
    sort { XmosBuildLib::ByTarget($a, $b) }
      (@{ $xmosArg->GetTargets() });

  $MAKE_FLAGS = $xmosArg->GetMakeFlags();
  $CONFIG = $xmosArg->GetOption("CONFIG");
  $DOMAIN = $xmosArg->GetOption("DOMAIN");
  $HOST = $xmosArg->GetOption("HOST");
  $BIN = $xmosArg->GetBinDir();

  foreach my $target (@targets) {
    DoTarget($target);
  }
  return 0;
}

sub DoTarget
{
  my $target = $_[0];
  if ($target eq "list_targets") {
    ListTargets();
  } else {
    my $targets = $ALIASES{$target};
    if (defined($targets)) {
      # Target is an alias
      foreach my $target (@$targets) {
        DoTarget($target);
      }
    } else {
      my $function = $TARGETS{$target}[0];
      if (defined($function)) {
        print(" ++ $target\n");
        &$function();
      }
    }
  }
}

sub ListTargets
{
  foreach my $target (keys(%TARGETS)) {
    print("$target\n");
  }
  foreach my $alias (keys(%ALIASES)) {
    print("$alias\n");
  }
}

sub DoBuild
{
  chdir("${XMOS_ROOT}${SLASH}ai_tools");
  system("make submodule_update") == 0
    or die "Failed to configure submodules";
  chdir("experimental/xformer");
  if ($^O eq "linux") {
    $ENV{'JAVA_HOME'} = '/usr/lib/jvm/java-1.8.0';
  }
  system("bazel build //:xcore-opt") == 0
    or die "Failed to build xformer-2.0";
}

sub DoClean
{
  chdir("${XMOS_ROOT}${SLASH}ai_tools");
  chdir("experimental${SLASH}xformer");
  File::Path::rmtree('bazel-bin');
  File::Path::rmtree('bazel-out');
  File::Path::rmtree('bazel-testlogs');
  File::Path::rmtree('bazel-xformer');
}

sub DoInstall
{
  chdir("${XMOS_ROOT}${SLASH}ai_tools");
  XmosBuildLib::InstallReleaseDirectory($DOMAIN, "experimental/xformer/bazel-bin", "xcore-opt");
}

main()
