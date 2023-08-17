# NEW
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")
# NEW
load(
   "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
   "feature",
   "flag_group",
   "flag_set",
   "tool_path",
)

all_link_actions = [ # NEW
   ACTION_NAMES.cpp_link_executable,
   ACTION_NAMES.cpp_link_dynamic_library,
   ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

def _impl(ctx):
   tool_paths = [ # NEW
       tool_path(
           name = "gcc",
           path = "/opt/xmos/gcc/11.2.0/bin/gcc",
       ),
       tool_path(
           name = "ld",
           path = "/usr/bin/ld",
       ),
       tool_path(
           name = "ar",
           path = "/opt/xmos/gcc/11.2.0/bin/gcc-ar",
       ),
       tool_path(
           name = "cpp",
           path = "/opt/xmos/gcc/11.2.0/bin/cpp",
       ),
       tool_path(
           name = "gcov",
           path = "/bin/false",
       ),
       tool_path(
           name = "nm",
           path = "/bin/false",
       ),
       tool_path(
           name = "objdump",
           path = "/bin/false",
       ),
       tool_path(
           name = "strip",
           path = "/bin/false",
       ),
   ]

   features = [ # NEW
       feature(
	   name = "default_linker_flags",
	   enabled = True,
	   flag_sets = [
	       flag_set(
		   actions = all_link_actions,
		   flag_groups = ([
		       flag_group(
			   flags = [
			       "-lstdc++",
			   ],
		       ),
		   ]),
	       ),
	   ],
       ),
   ]

   return cc_common.create_cc_toolchain_config_info(
       ctx = ctx,
       features = features, # NEW
       cxx_builtin_include_directories = [
	   "/opt/xmos/gcc/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/include",
	   "/opt/xmos/gcc/11.2.0/lib/gcc/x86_64-pc-linux-gnu/11.2.0/include-fixed",
	   "/opt/xmos/gcc/11.2.0/include",
	   "/usr/include",
       ],
       toolchain_identifier = "local",
       host_system_name = "local",
       target_system_name = "local",
       target_cpu = "k8",
       target_libc = "unknown",
       compiler = "gnu",
       abi_version = "unknown",
       abi_libc_version = "unknown",
       tool_paths = tool_paths
   )

cc_toolchain_config = rule(
   implementation = _impl,
   attrs = {},
   provides = [CcToolchainConfigInfo],
)
