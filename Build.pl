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
my $BAZEL_VERSION;
my $PTH;
my $BAZELCONFIG;

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

  if ($^O eq "linux") {
    $ENV{"JAVA_HOME"} = "/usr/lib/jvm/java-1.8.0";
    $ENV{"CC"} = "/opt/xmos/gcc/11.2.0/bin/gcc";
  } elsif ($^O eq "darwin") {
    $BAZEL_VERSION="4.1.0";
    system("curl -fLO 'https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh'");
    system("chmod +x 'bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh'");
    system("./bazel-${BAZEL_VERSION}-installer-darwin-x86_64.sh --prefix=${XMOS_ROOT}${SLASH}bazel");
    $PTH=$ENV{"PATH"};
    $ENV{"PATH"} = "$PTH:${XMOS_ROOT}${SLASH}bazel";
    $BAZELCONFIG="--config=darwin_config";
  } elsif ($^O eq "MSWin32") {
    ;
  } else {
    die "Os $^O not recognised";
  }
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
  #system("make submodule_update") == 0                                         or die "Failed to configure submodules";
  system("python3 -m venv .venv") == 0                                         or die "Python -m venv failed";
  system(". .venv/bin/activate && pip install -r requirements.txt") == 0 or die "Python requirement install failed";
  system(". .venv/bin/activate && cd experimental${SLASH}xformer && bazel build $BAZELCONFIG --remote_cache=http://srv-bri-bld-cache:8080 //:xcore-opt --verbose_failures") == 0              or die "Failed to build xformer-2.0";
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
  system("mv experimental/xformer/bazel-bin/xcore-opt experimental/xformer/bazel-bin/xformer");
  XmosBuildLib::InstallReleaseDirectory($DOMAIN, "experimental/xformer/bazel-bin", "xformer");
}

main()
