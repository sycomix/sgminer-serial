/*
 * Copyright 2011-2012 Con Kolivas
 * Copyright 2011-2012 Luke Dashjr
 * Copyright 2010 Jeff Garzik
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.  See COPYING for more details.
 */

#include "config.h"

#ifdef HAVE_CURSES
#include <curses.h>
#endif

#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <signal.h>
#include <sys/types.h>

#ifndef WIN32
#include <sys/resource.h>
#endif
#include <ccan/opt/opt.h>

#include "compat.h"
#include "miner.h"
#include "config_parser.h"
#include "driver-opencl.h"
#include "findnonce.h"
#include "ocl.h"
#include "adl.h"
#include "util.h"
#include "sysfs-gpu-controls.h"

#include "algorithm/equihash.h"

/* TODO: cleanup externals ********************/

#ifdef HAVE_CURSES
extern WINDOW *mainwin, *statuswin, *logwin;
extern void enable_curses(void);
#endif

extern int mining_threads;
extern double total_secs;
extern int opt_g_threads;
extern bool opt_loginput;
extern char *opt_kernel_path;
extern int gpur_thr_id;
extern bool opt_noadl;

extern void *miner_thread(void *userdata);
extern int dev_from_id(int thr_id);
extern void decay_time(double *f, double fadd);

/**********************************************/

char *set_vector(char *arg)
{
  int i, val = 0, device = 0;
  char *nextptr;

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set vector";
  val = atoi(nextptr);
  if (val != 1 && val != 2 && val != 4)
    return "Invalid value passed to set_vector";

  gpus[device++].vwidth = val;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);
    if (val != 1 && val != 2 && val != 4)
      return "Invalid value passed to set_vector";

    gpus[device++].vwidth = val;
  }
  if (device == 1) {
    for (i = device; i < MAX_GPUDEVICES; i++)
      gpus[i].vwidth = gpus[0].vwidth;
  }

  return NULL;
}

char *set_worksize(const char *arg)
{
  int i, val = 0, device = 0;
  char *tmpstr = strdup(arg);
  char *nextptr;

  if ((nextptr = strtok(tmpstr, ",")) == NULL) {
    free(tmpstr);
    return "Invalid parameters for set work size";
  }

  do {
    val = atoi(nextptr);

    if (val < 1 || val > 9999) {
      free(tmpstr);
      return "Invalid value passed to set_worksize";
    }

    applog(LOG_DEBUG, "GPU %d Worksize set to %u.", device, val);
    gpus[device++].work_size = val;
  } while ((nextptr = strtok(NULL, ",")) != NULL);

  // if only 1 worksize was passed, assign the same worksize for all remaining GPUs
  if (device == 1) {
    for (i = device; i < total_devices; ++i) {
      gpus[i].work_size = gpus[0].work_size;
      applog(LOG_DEBUG, "GPU %d Worksize set to %u.", i, gpus[i].work_size);
    }
  }

  free(tmpstr);
  return NULL;
}

char *set_shaders(char *arg)
{
  int i, val = 0, device = 0;
  char *nextptr;

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set lookup gap";
  val = atoi(nextptr);

  gpus[device++].shaders = val;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);

    gpus[device++].shaders = val;
  }
  if (device == 1) {
    for (i = device; i < MAX_GPUDEVICES; i++)
      gpus[i].shaders = gpus[0].shaders;
  }

  return NULL;
}

char *set_lookup_gap(char *arg)
{
  int i, val = 0, device = 0;
  char *nextptr;

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set lookup gap";
  val = atoi(nextptr);

  gpus[device++].opt_lg = val;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);

    gpus[device++].opt_lg = val;
  }
  if (device == 1) {
    for (i = device; i < MAX_GPUDEVICES; i++)
      gpus[i].opt_lg = gpus[0].opt_lg;
  }

  return NULL;
}

char *set_thread_concurrency(const char *arg)
{
  int i, device = 0;
  size_t val = 0;
  char *tmpstr = strdup(arg);
  char *nextptr;

  // empty string - use 0 and let algo autodetect the TC
  if (empty_string(tmpstr)) {
    applog(LOG_DEBUG, "GPU %d Thread Concurrency set to %lu.", device, val);
    gpus[device++].opt_tc = val;
  }
  // not empty string
  else {
    if ((nextptr = strtok(tmpstr, ",")) == NULL) {
      free(tmpstr);
      return "Invalid parameters for set_thread_concurrency";
    }

    do {
      val = (unsigned long)atol(nextptr);

      applog(LOG_DEBUG, "GPU %d Thread Concurrency set to %lu.", device, val);
      gpus[device++].opt_tc = val;
    } while ((nextptr = strtok(NULL, ",")) != NULL);
  }

  // if only 1 TC was passed, assign the same worksize for all remaining GPUs
  if (device == 1) {
    for (i = device; i < total_devices; ++i) {
      gpus[i].opt_tc = gpus[0].opt_tc;
      applog(LOG_DEBUG, "GPU %d Thread Concurrency set to %lu.", i, gpus[i].opt_tc);
    }
  }

  free(tmpstr);
  return NULL;
}

/* This function allows us to map an adl device to an opencl device for when
 * simple enumeration has failed to match them. */
char *set_gpu_map(char *arg)
{
  int val1 = 0, val2 = 0;
  char *nextptr;

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set gpu map";
  if (sscanf(arg, "%d:%d", &val1, &val2) != 2)
    return "Invalid description for map pair";
  if (val1 < 0 || val1 > MAX_GPUDEVICES || val2 < 0 || val2 > MAX_GPUDEVICES)
    return "Invalid value passed to set_gpu_map";

  gpus[val1].virtual_adl = val2;
  gpus[val1].mapped = true;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    if (sscanf(nextptr, "%d:%d", &val1, &val2) != 2)
      return "Invalid description for map pair";
    if (val1 < 0 || val1 > MAX_GPUDEVICES || val2 < 0 || val2 > MAX_GPUDEVICES)
      return "Invalid value passed to set_gpu_map";
    gpus[val1].virtual_adl = val2;
    gpus[val1].mapped = true;
  }

  return NULL;
}

char *set_gpu_threads(const char *_arg)
{
  int i, val = 1, device = 0;
  char *nextptr;
  char *arg = (char *)alloca(strlen(_arg) + 1);
  strcpy(arg, _arg);

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set_gpu_threads";
  val = atoi(nextptr);
  if (val < 1 || val > 20) // gpu_threads increase max value to 20
    return "Invalid value passed to set_gpu_threads";

  gpus[device++].threads = val;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);
    if (val < 1 || val > 20) // gpu_threads increase max value to 20
      return "Invalid value passed to set_gpu_threads";

    gpus[device++].threads = val;
  }
  if (device == 1) {
    for (i = device; i < MAX_GPUDEVICES; i++)
      gpus[i].threads = gpus[0].threads;
  }

  return NULL;
}

char *set_gpu_engine(const char *_arg)
{
  int i, val1 = 0, val2 = 0, device = 0;
  char *nextptr;
  char *arg = (char *)alloca(strlen(_arg) + 1);
  strcpy(arg, _arg);

  if (!(nextptr = strtok(arg, ",")))
    return "Invalid parameters for set gpu engine";

  do {
    get_intrange(nextptr, &val1, &val2);
    if (val1 < 0 || val1 > 9999 || val2 < 0 || val2 > 9999)
      return "Invalid value passed to set_gpu_engine";

    gpus[device].min_engine = val1;
    gpus[device].gpu_engine = val2;

    //also set adl settings otherwise range will never properly be applied
    //since min_engine/gpu_engine are only called during init_adl() at startup
    gpus[device].adl.minspeed = val1 * 100;
    gpus[device].adl.maxspeed = val2 * 100;

    device++;
  } while ((nextptr = strtok(NULL, ",")) != NULL);

  //if only 1 range passed, apply to all gpus
  if (device == 1) {
    for (i = 1; i < MAX_GPUDEVICES; i++) {
      gpus[i].min_engine = gpus[0].min_engine;
      gpus[i].gpu_engine = gpus[0].gpu_engine;

      //set adl values
      gpus[i].adl.minspeed = val1 * 100;
      gpus[i].adl.maxspeed = val2 * 100;
    }
  }

  return NULL;
}

char *set_gpu_fan(const char *_arg)
{
  int i, val1 = 0, val2 = 0, device = 0;
  char *nextptr;
  char *arg = (char *)alloca(strlen(_arg) + 1);
  strcpy(arg, _arg);

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set gpu fan";
  get_intrange(nextptr, &val1, &val2);
  if (val1 < 0 || val1 > 100 || val2 < 0 || val2 > 100)
    return "Invalid value passed to set_gpu_fan";

  gpus[device].min_fan = val1;
  gpus[device].gpu_fan = val2;
  device++;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    get_intrange(nextptr, &val1, &val2);
    if (val1 < 0 || val1 > 100 || val2 < 0 || val2 > 100)
      return "Invalid value passed to set_gpu_fan";

    gpus[device].min_fan = val1;
    gpus[device].gpu_fan = val2;
    device++;
  }

  if (device == 1) {
    for (i = 1; i < MAX_GPUDEVICES; i++) {
      gpus[i].min_fan = gpus[0].min_fan;
      gpus[i].gpu_fan = gpus[0].gpu_fan;
    }
  }

  return NULL;
}

char *set_gpu_memclock(const char *_arg)
{
  int i, val = 0, device = 0;
  char *nextptr;
  char *arg = (char *)alloca(strlen(_arg) + 1);
  strcpy(arg, _arg);

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set gpu memclock";
  val = atoi(nextptr);
  if (val < 0 || val >= 9999)
    return "Invalid value passed to set_gpu_memclock";

  gpus[device++].gpu_memclock = val;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);
    if (val < 0 || val >= 9999)
      return "Invalid value passed to set_gpu_memclock";

    gpus[device++].gpu_memclock = val;
  }
  if (device == 1) {
    for (i = device; i < MAX_GPUDEVICES; i++)
      gpus[i].gpu_memclock = gpus[0].gpu_memclock;
  }

  return NULL;
}

char *set_gpu_memdiff(char *arg)
{
  int i, val = 0, device = 0;
  char *nextptr;

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set gpu memdiff";
  val = atoi(nextptr);
  if (val < -9999 || val > 9999)
    return "Invalid value passed to set_gpu_memdiff";

  gpus[device++].gpu_memdiff = val;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);
    if (val < -9999 || val > 9999)
      return "Invalid value passed to set_gpu_memdiff";

    gpus[device++].gpu_memdiff = val;
  }
  if (device == 1) {
    for (i = device; i < MAX_GPUDEVICES; i++)
      gpus[i].gpu_memdiff = gpus[0].gpu_memdiff;
  }

  return NULL;
}

char *set_gpu_powertune(char *arg)
{
  int i, val = 0, device = 0;
  char *nextptr;

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set gpu powertune";
  val = atoi(nextptr);
  if (val < -99 || val > 99)
    return "Invalid value passed to set_gpu_powertune";

  gpus[device++].gpu_powertune = val;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);
    if (val < -99 || val > 99)
      return "Invalid value passed to set_gpu_powertune";

    gpus[device++].gpu_powertune = val;
  }
  if (device == 1) {
    for (i = device; i < MAX_GPUDEVICES; i++)
      gpus[i].gpu_powertune = gpus[0].gpu_powertune;
  }

  return NULL;
}

char *set_gpu_vddc(char *arg)
{
  int i, device = 0;
  float val = 0;
  char *nextptr;

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set gpu vddc";
  val = atof(nextptr);
  if (val < 0 || val >= 9999)
    return "Invalid value passed to set_gpu_vddc";

  gpus[device++].gpu_vddc = val;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atof(nextptr);
    if (val < 0 || val >= 9999)
      return "Invalid value passed to set_gpu_vddc";

    gpus[device++].gpu_vddc = val;
  }
  if (device == 1) {
    for (i = device; i < MAX_GPUDEVICES; i++)
      gpus[i].gpu_vddc = gpus[0].gpu_vddc;
  }

  return NULL;
}

char *set_temp_overheat(char *arg)
{
  int i, val = 0, device = 0, *to;
  char *nextptr;

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set temp overheat";
  val = atoi(nextptr);
  if (val < 0 || val > 200)
    return "Invalid value passed to set temp overheat";

  gpus[device].adl.overtemp = val;
  gpus[device++].sysfs_info.OverHeatTemp = val;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);
    if (val < 0 || val > 200)
      return "Invalid value passed to set temp overheat";

    gpus[device].adl.overtemp = val;
    gpus[device++].sysfs_info.OverHeatTemp = val;
  }
  if (device == 1) {
    for (i = device; i < MAX_GPUDEVICES; i++) {
      gpus[i].adl.overtemp = val;
      gpus[i].sysfs_info.OverHeatTemp = val;
    }
  }

  return NULL;
}

char *set_temp_target(char *arg)
{
  int i, val = 0, device = 0, *tt;
  char *nextptr;

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set temp target";
  val = atoi(nextptr);
  if (val < 0 || val > 200)
    return "Invalid value passed to set temp target";

  tt = &gpus[device].adl.targettemp;
  *tt = val;
  tt = (int*)&gpus[device++].sysfs_info.TargetTemp;
  *tt = val;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);
    if (val < 0 || val > 200)
      return "Invalid value passed to set temp target";

    tt = &gpus[device].adl.targettemp;
    *tt = val;
    tt = (int*)&gpus[device++].sysfs_info.TargetTemp;
    *tt = val;    
  }
  if (device == 1) {
    for (i = device; i < MAX_GPUDEVICES; i++) {
      tt = &gpus[i].adl.targettemp;
      *tt = val;
      tt = (int*)&gpus[i].sysfs_info.TargetTemp;
      *tt = val;
    }
  }

  return NULL;
}

char *set_intensity(const char *_arg)
{
  int i, device = 0, *tt;
  char *nextptr, val = 0;
  char *arg = (char *)alloca(strlen(_arg) + 1);
  strcpy(arg, _arg);

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for set intensity";
  if (!strncasecmp(nextptr, "d", 1))
    gpus[device].dynamic = true;
  else {
    gpus[device].dynamic = false;
    val = atoi(nextptr);
    if (val == 0) return "disabled";
    if (val < MIN_INTENSITY || val > MAX_INTENSITY)
      return "Invalid value passed to set intensity";
    tt = &gpus[device].intensity;
    *tt = val;
    gpus[device].xintensity = 0; // Disable shader based intensity
    gpus[device].rawintensity = 0; // Disable raw intensity
  }

  device++;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    if (!strncasecmp(nextptr, "d", 1))
      gpus[device].dynamic = true;
    else {
      gpus[device].dynamic = false;
      val = atoi(nextptr);
      if (val == 0) return "disabled";
      if (val < MIN_INTENSITY || val > MAX_INTENSITY)
        return "Invalid value passed to set intensity";

      tt = &gpus[device].intensity;
      *tt = val;
      gpus[device].xintensity = 0; // Disable shader based intensity
      gpus[device].rawintensity = 0; // Disable raw intensity
    }
    device++;
  }
  if (device == 1) {
    for (i = device; i < MAX_GPUDEVICES; i++) {
      gpus[i].dynamic = gpus[0].dynamic;
      gpus[i].intensity = gpus[0].intensity;
      gpus[i].xintensity = 0; // Disable shader based intensity
      gpus[i].rawintensity = 0; // Disable raw intensity
    }
  }

  return NULL;
}

char *set_xintensity(const char *_arg)
{
  int i, device = 0, val = 0;
  char *nextptr;
  char *arg = (char *)alloca(strlen(_arg) + 1);
  strcpy(arg, _arg);

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for shader based intensity";
  val = atoi(nextptr);
  if (val == 0) return "disabled";
  if (val < MIN_XINTENSITY || val > MAX_XINTENSITY)
    return "Invalid value passed to set shader-based intensity";

  gpus[device].dynamic = false; // Disable dynamic intensity
  gpus[device].intensity = 0; // Disable regular intensity
  gpus[device].rawintensity = 0; // Disable raw intensity
  gpus[device].xintensity = val;
  device++;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);
    if (val == 0) return "disabled";
    if (val < MIN_XINTENSITY || val > MAX_XINTENSITY)
      return "Invalid value passed to set shader based intensity";
    gpus[device].dynamic = false; // Disable dynamic intensity
    gpus[device].intensity = 0; // Disable regular intensity
    gpus[device].rawintensity = 0; // Disable raw intensity
    gpus[device].xintensity = val;
    device++;
  }
  if (device == 1)
  for (i = device; i < MAX_GPUDEVICES; i++) {
    gpus[i].dynamic = gpus[0].dynamic;
    gpus[i].intensity = gpus[0].intensity;
    gpus[i].rawintensity = gpus[0].rawintensity;
    gpus[i].xintensity = gpus[0].xintensity;
  }

  return NULL;
}

char *set_rawintensity(const char *_arg)
{
  int i, device = 0, val = 0;
  char *nextptr;
  char *arg = (char *)alloca(strlen(_arg) + 1);
  strcpy(arg, _arg);

  nextptr = strtok(arg, ",");
  if (nextptr == NULL)
    return "Invalid parameters for raw intensity";
  val = atoi(nextptr);
  if (val == 0) return "disabled";
  if (val < MIN_RAWINTENSITY || val > MAX_RAWINTENSITY)
    return "Invalid value passed to set raw intensity";

  gpus[device].dynamic = false; // Disable dynamic intensity
  gpus[device].intensity = 0; // Disable regular intensity
  gpus[device].xintensity = 0; // Disable xintensity
  gpus[device].rawintensity = val;
  device++;

  while ((nextptr = strtok(NULL, ",")) != NULL) {
    val = atoi(nextptr);
    if (val == 0) return "disabled";
    if (val < MIN_RAWINTENSITY || val > MAX_RAWINTENSITY)
      return "Invalid value passed to set raw intensity";
    gpus[device].dynamic = false; // Disable dynamic intensity
    gpus[device].intensity = 0; // Disable regular intensity
    gpus[device].xintensity = 0; // Disable xintensity
    gpus[device].rawintensity = val;
    device++;
  }
  if (device == 1)
  for (i = device; i < MAX_GPUDEVICES; i++) {
    gpus[i].dynamic = gpus[0].dynamic;
    gpus[i].intensity = gpus[0].intensity;
    gpus[i].rawintensity = gpus[0].rawintensity;
    gpus[i].xintensity = gpus[0].xintensity;
  }

  return NULL;
}

void print_ndevs(int *ndevs)
{
  opt_verbose = true;
  opencl_drv.drv_detect();
  clear_adl(*ndevs);
  applog(LOG_INFO, "%i GPU devices max detected", *ndevs);
}

struct cgpu_info gpus[MAX_GPUDEVICES]; /* Maximum number apparently possible */
struct cgpu_info *cpus;

/* In dynamic mode, only the first thread of each device will be in use.
 * This potentially could start a thread that was stopped with the start-stop
 * options if one were to disable dynamic from the menu on a paused GPU */
void pause_dynamic_threads(int gpu)
{
  struct cgpu_info *cgpu = &gpus[gpu];
  int i;

  rd_lock(&mining_thr_lock); 
  for (i = 1; i < cgpu->threads; i++) {
    struct thr_info *thr;

    thr = cgpu->thr[i];
    if (!thr->pause && cgpu->dynamic) {
      applog(LOG_WARNING, "Disabling extra threads due to dynamic mode.");
      applog(LOG_WARNING, "Tune dynamic intensity with --gpu-dyninterval");
    }

    thr->pause = cgpu->dynamic;
    if (!cgpu->dynamic && cgpu->deven != DEV_DISABLED)
      cgsem_post(&thr->sem);
  }
  rd_unlock(&mining_thr_lock);
}

#if defined(HAVE_CURSES)
void manage_gpu(void)
{
  struct thr_info *thr;
  int selected, gpu, i;
  char checkin[40];
  char input;

  if (!opt_g_threads) {
    applog(LOG_ERR, "opt_g_threads not set in manage_gpu()");
    return;
  }

  opt_loginput = true;
  immedok(logwin, true);
  clear_logwin();
retry: // TODO: refactor

  for (gpu = 0; gpu < nDevs; gpu++) {
    struct cgpu_info *cgpu = &gpus[gpu];
    double displayed_rolling, displayed_total;
    bool mhash_base = true;

    displayed_rolling = cgpu->rolling;
    displayed_total = cgpu->total_mhashes / total_secs;
    if (displayed_rolling < 1) {
      displayed_rolling *= 1000;
      displayed_total *= 1000;
      mhash_base = false;
    }

    wlog("GPU %d: %.1f / %.1f %sh/s | A:%d  R:%d  HW:%d  U:%.2f/m  I:%d  xI:%d  rI:%d\n",
      gpu, displayed_rolling, displayed_total, mhash_base ? "M" : "K",
      cgpu->accepted, cgpu->rejected, cgpu->hw_errors,
      cgpu->utility, cgpu->intensity, cgpu->xintensity, cgpu->rawintensity);

    if (gpus[gpu].has_adl || gpus[gpu].has_sysfs_hwcontrols) {
      int engineclock = 0, memclock = 0, activity = 0, fanspeed = 0, fanpercent = 0, powertune = 0;
      float temp = 0, vddc = 0;

      if (gpu_stats(gpu, &temp, &engineclock, &memclock, &vddc, &activity, &fanspeed, &fanpercent, &powertune)) {
        char logline[255];

        strcpy(logline, ""); // In case it has no data
        if (temp != -1)
          sprintf(logline, "%.1f C  ", temp);
        if (fanspeed != -1 || fanpercent != -1) {
          tailsprintf(logline, sizeof(logline), "F: ");
          if (fanpercent != -1)
            tailsprintf(logline, sizeof(logline), "%d%% ", fanpercent);
          if (fanspeed != -1)
            tailsprintf(logline, sizeof(logline), "(%d RPM) ", fanspeed);
          tailsprintf(logline, sizeof(logline), " ");
        }
        if (engineclock != -1)
          tailsprintf(logline, sizeof(logline), "E: %d MHz  ", engineclock);
        if (memclock != -1)
          tailsprintf(logline, sizeof(logline), "M: %d Mhz  ", memclock);
        if (vddc != -1)
          tailsprintf(logline, sizeof(logline), "V: %.3fV  ", vddc);
        if (activity != -1)
          tailsprintf(logline, sizeof(logline), "A: %d%%  ", activity);
        if (powertune != -1)
          tailsprintf(logline, sizeof(logline), "P: %d%%", powertune);
        tailsprintf(logline, sizeof(logline), "\n");
        _wlog(logline);
      }
    }
    wlog("Last initialised: %s\n", cgpu->init);

    rd_lock(&mining_thr_lock);
    for (i = 0; i < mining_threads; i++) {
      thr = mining_thr[i];
      if (thr->cgpu != cgpu)
        continue;
      get_datestamp(checkin, sizeof(checkin), &thr->last);
      displayed_rolling = thr->rolling;
      if (!mhash_base)
        displayed_rolling *= 1000;
      wlog("Thread %d: %.1f %sh/s %s ", i, displayed_rolling, mhash_base ? "M" : "K", cgpu->deven != DEV_DISABLED ? "Enabled" : "Disabled");
      switch (cgpu->status) {
      default:
      case LIFE_WELL:
        wlog("ALIVE");
        break;
      case LIFE_SICK:
        wlog("SICK reported in %s", checkin);
        break;
      case LIFE_DEAD:
        wlog("DEAD reported in %s", checkin);
        break;
      case LIFE_INIT:
      case LIFE_NOSTART:
        wlog("Never started");
        break;
      }
      if (thr->pause)
        wlog(" paused");
      wlog("\n");
    }
    rd_unlock(&mining_thr_lock);

    wlog("\n");
  }

  wlogprint("[E]nable  [D]isable  [R]estart GPU  %s\n", adl_active ? "[C]hange settings" : "");
  wlogprint("[I]ntensity  E[x]perimental intensity  R[a]w Intensity\n");

  wlogprint("Or press any other key to continue\n");
  logwin_update();
  input = getch();

  if (nDevs == 1)
    selected = 0;
  else
    selected = -1;
  if (!strncasecmp(&input, "e", 1)) {
    struct cgpu_info *cgpu;

    if (selected)
      selected = curses_int("Select GPU to enable");
    if (selected < 0 || selected >= nDevs) {
      wlogprint("Invalid selection\n");
      goto retry;
    }
    if (gpus[selected].deven != DEV_DISABLED) {
      wlogprint("Device already enabled\n");
      goto retry;
    }
    gpus[selected].deven = DEV_ENABLED;
    rd_lock(&mining_thr_lock);
    for (i = 0; i < mining_threads; ++i) {
      thr = mining_thr[i];
      cgpu = thr->cgpu;
      if (cgpu->drv->drv_id != DRIVER_opencl)
        continue;
      if (dev_from_id(i) != selected)
        continue;
      if (cgpu->status != LIFE_WELL) {
        wlogprint("Must restart device before enabling it");
        goto retry;
      }
      applog(LOG_DEBUG, "Pushing sem post to thread %d", thr->id);

      cgsem_post(&thr->sem);
    }
    rd_unlock(&mining_thr_lock);
    goto retry;
  }
  else if (!strncasecmp(&input, "d", 1)) {
    if (selected)
      selected = curses_int("Select GPU to disable");
    if (selected < 0 || selected >= nDevs) {
      wlogprint("Invalid selection\n");
      goto retry;
    }
    if (gpus[selected].deven == DEV_DISABLED) {
      wlogprint("Device already disabled\n");
      goto retry;
    }
    gpus[selected].deven = DEV_DISABLED;
    goto retry;
  }
  else if (!strncasecmp(&input, "i", 1)) {
    int intensity;
    char *intvar;

    if (selected)
      selected = curses_int("Select GPU to change intensity on");
    if (selected < 0 || selected >= nDevs) {
      wlogprint("Invalid selection\n");
      goto retry;
    }

    intvar = curses_input("Set GPU scan intensity (d or "
      MIN_INTENSITY_STR " -> "
      MAX_INTENSITY_STR ")");
    if (!intvar) {
      wlogprint("Invalid input\n");
      goto retry;
    }
    if (!strncasecmp(intvar, "d", 1)) {
      wlogprint("Dynamic mode enabled on gpu %d\n", selected);
      gpus[selected].dynamic = true;

      // fix config with new settings so that we can save them
      update_config_intensity(get_gpu_profile(selected));

      pause_dynamic_threads(selected);
      free(intvar);
      goto retry;
    }
    intensity = atoi(intvar);
    free(intvar);
    if (intensity < MIN_INTENSITY || intensity > MAX_INTENSITY) {
      wlogprint("Invalid selection\n");
      goto retry;
    }
    gpus[selected].dynamic = false;
    gpus[selected].intensity = intensity;
    gpus[selected].xintensity = 0; // Disable xintensity when enabling intensity
    gpus[selected].rawintensity = 0; // Disable raw intensity when enabling intensity
    wlogprint("Intensity on gpu %d set to %d\n", selected, intensity);

    // fix config with new settings so that we can save them
    update_config_intensity(get_gpu_profile(selected));

    pause_dynamic_threads(selected);
    goto retry;
  }
  else if (!strncasecmp(&input, "x", 1)) {
    int xintensity;
    char *intvar;

    if (selected)
      selected = curses_int("Select GPU to change experimental intensity on");
    if (selected < 0 || selected >= nDevs) {
      wlogprint("Invalid selection\n");
      goto retry;
    }

    intvar = curses_input("Set experimental GPU scan intensity (" MIN_XINTENSITY_STR " -> " MAX_XINTENSITY_STR ")");
    if (!intvar) {
      wlogprint("Invalid input\n");
      goto retry;
    }
    xintensity = atoi(intvar);
    free(intvar);
    if (xintensity < MIN_XINTENSITY || xintensity > MAX_XINTENSITY) {
      wlogprint("Invalid selection\n");
      goto retry;
    }
    gpus[selected].dynamic = false;
    gpus[selected].intensity = 0; // Disable intensity when enabling xintensity
    gpus[selected].rawintensity = 0; // Disable raw intensity when enabling xintensity
    gpus[selected].xintensity = xintensity;
    wlogprint("Experimental intensity on gpu %d set to %d\n", selected, xintensity);

    // fix config with new settings so that we can save them
    update_config_xintensity(get_gpu_profile(selected));

    pause_dynamic_threads(selected);
    goto retry;
  }
  else if (!strncasecmp(&input, "a", 1)) {
    int rawintensity;
    char *intvar;

    if (selected)
      selected = curses_int("Select GPU to change raw intensity on");
    if (selected < 0 || selected >= nDevs) {
      wlogprint("Invalid selection\n");
      goto retry;
    }

    intvar = curses_input("Set raw GPU scan intensity (" MIN_RAWINTENSITY_STR " -> " MAX_RAWINTENSITY_STR ")");
    if (!intvar) {
      wlogprint("Invalid input\n");
      goto retry;
    }
    rawintensity = atoi(intvar);
    free(intvar);
    if (rawintensity < MIN_RAWINTENSITY || rawintensity > MAX_RAWINTENSITY) {
      wlogprint("Invalid selection\n");
      goto retry;
    }
    gpus[selected].dynamic = false;
    gpus[selected].intensity = 0; // Disable intensity when enabling raw intensity
    gpus[selected].xintensity = 0; // Disable xintensity when enabling raw intensity
    gpus[selected].rawintensity = rawintensity;
    wlogprint("Raw intensity on gpu %d set to %d\n", selected, rawintensity);

    // fix config with new settings so that we can save them
    update_config_rawintensity(get_gpu_profile(selected));

    pause_dynamic_threads(selected);
    goto retry;
  }
  else if (!strncasecmp(&input, "r", 1)) {
    if (selected)
      selected = curses_int("Select GPU to attempt to restart");
    if (selected < 0 || selected >= nDevs) {
      wlogprint("Invalid selection\n");
      goto retry;
    }
    wlogprint("Attempting to restart threads of GPU %d\n", selected);
    reinit_device(&gpus[selected]);
    goto retry;
  }
  else if (adl_active && (!strncasecmp(&input, "c", 1))) {
    if (selected)
      selected = curses_int("Select GPU to change settings on");
    if (selected < 0 || selected >= nDevs) {
      wlogprint("Invalid selection\n");
      goto retry;
    }
    change_gpusettings(selected);
    goto retry;
  }
  else
    clear_logwin();

  immedok(logwin, false);
  opt_loginput = false;
}
#else
void manage_gpu(void)
{
}
#endif

static _clState *clStates[MAX_GPUDEVICES];

static void set_threads_hashes(unsigned int vectors, unsigned int compute_shaders, int64_t *hashes, size_t *globalThreads,
  unsigned int minthreads, __maybe_unused int *intensity, __maybe_unused int *xintensity,
  __maybe_unused int *rawintensity, algorithm_t *algorithm)
{
  unsigned int threads = 0;
  while (threads < minthreads) {

    if (*rawintensity > 0) {
      threads = *rawintensity;
    }
    else if (*xintensity > 0) {
      threads = compute_shaders * ((algorithm->xintensity_shift) ? (1 << (algorithm->xintensity_shift + *xintensity)) : *xintensity);
    }
    else {
      threads = 1 << (algorithm->intensity_shift + *intensity);
    }

    if (threads < minthreads) {
      if (likely(*intensity < MAX_INTENSITY)) {
        (*intensity)++;
      }
      else {
        threads = minthreads;
      }
    }
  }

  *globalThreads = threads;
  *hashes = threads * vectors;
}

/* We have only one thread that ever re-initialises GPUs, thus if any GPU
 * init command fails due to a completely wedged GPU, the thread will never
 * return, unable to harm other GPUs. If it does return, it means we only had
 * a soft failure and then the reinit_gpu thread is ready to tackle another
 * GPU */
void *reinit_gpu(void *userdata)
{
  struct thr_info *mythr = (struct thr_info *)userdata;
  struct cgpu_info *cgpu;
  struct thr_info *thr;
  struct timeval now;
  char name[256];
  int thr_id;
  int gpu;

  pthread_detach(pthread_self());

select_cgpu:
  cgpu = (struct cgpu_info *)tq_pop(mythr->q, NULL);
  if (!cgpu)
    goto out;

  if (clDevicesNum() != nDevs) {
    applog(LOG_WARNING, "Hardware not reporting same number of active devices, will not attempt to restart GPU");
    goto out;
  }

  gpu = cgpu->device_id;

  rd_lock(&mining_thr_lock);
  for (thr_id = 0; thr_id < mining_threads; ++thr_id) {
    thr = mining_thr[thr_id];
    cgpu = thr->cgpu;
    if (cgpu->drv->drv_id != DRIVER_opencl)
      continue;
    if (dev_from_id(thr_id) != gpu)
      continue;

    thr->rolling = thr->cgpu->rolling = 0;
    /* Reports the last time we tried to revive a sick GPU */
    cgtime(&thr->sick);
    if (!pthread_kill(thr->pth, 0)) {
      applog(LOG_WARNING, "Thread %d still exists, killing it off", thr_id);
      cg_completion_timeout(&thr_info_cancel_join, thr, 5000);
      thr->cgpu->drv->thread_shutdown(thr);
    }
    else
      applog(LOG_WARNING, "Thread %d no longer exists", thr_id);
  }
  rd_unlock(&mining_thr_lock);

  rd_lock(&mining_thr_lock);
  for (thr_id = 0; thr_id < mining_threads; ++thr_id) {
    int virtual_gpu;

    thr = mining_thr[thr_id];
    cgpu = thr->cgpu;
    if (cgpu->drv->drv_id != DRIVER_opencl)
      continue;
    if (dev_from_id(thr_id) != gpu)
      continue;

    virtual_gpu = cgpu->virtual_gpu;
    /* Lose this ram cause we may get stuck here! */
    //tq_freeze(thr->q);

    thr->q = tq_new();
    if (!thr->q)
      quit(1, "Failed to tq_new in reinit_gpu");

    /* Lose this ram cause we may dereference in the dying thread! */
    //free(clState);

    applog(LOG_INFO, "Reinit GPU thread %d", thr_id);
    clStates[thr_id] = initCl(virtual_gpu, name, sizeof(name), &cgpu->algorithm);
    if (!clStates[thr_id]) {
      applog(LOG_ERR, "Failed to reinit GPU thread %d", thr_id);
      goto select_cgpu;
    }
    applog(LOG_INFO, "initCl() finished. Found %s", name);

    if (unlikely(thr_info_create(thr, NULL, miner_thread, thr))) {
      applog(LOG_ERR, "thread %d create failed", thr_id);
      return NULL;
    }
    applog(LOG_WARNING, "Thread %d restarted", thr_id);
  }
  rd_unlock(&mining_thr_lock);

  cgtime(&now);
  get_datestamp(cgpu->init, sizeof(cgpu->init), &now);

  rd_lock(&mining_thr_lock);
  for (thr_id = 0; thr_id < mining_threads; ++thr_id) {
    thr = mining_thr[thr_id];
    cgpu = thr->cgpu;
    if (cgpu->drv->drv_id != DRIVER_opencl)
      continue;
    if (dev_from_id(thr_id) != gpu)
      continue;

    cgsem_post(&thr->sem);
  }
  rd_unlock(&mining_thr_lock);

  goto select_cgpu;
out:
  return NULL;
}

/*****************************************************************************************/

int serial_open(const char *devpath, unsigned long baud, signed short timeout, bool purge)
{
#ifdef WIN32
	HANDLE hSerial = CreateFile(devpath, GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, 0, NULL);

	if (unlikely(hSerial == INVALID_HANDLE_VALUE))
	{
		DWORD e = GetLastError();
		switch (e) {
		case ERROR_ACCESS_DENIED:
			applog(LOG_ERR, "Do not have user privileges required to open %s", devpath);
			break;
		case ERROR_SHARING_VIOLATION:
			applog(LOG_ERR, "%s is already in use by another process", devpath);
			break;
		case ERROR_FILE_NOT_FOUND:
			applog(LOG_ERR, "Device %s not found", devpath);
			break;
		default:
			applog(LOG_DEBUG, "Open %s failed, GetLastError:%d", devpath, (int)e);
			break;
		}
		return -1;
	}

	// thanks to af_newbie for pointers about this
	COMMCONFIG comCfg = { 0 };
	comCfg.dwSize = sizeof(COMMCONFIG);
	comCfg.wVersion = 1;
	comCfg.dcb.DCBlength = sizeof(DCB);
	comCfg.dcb.BaudRate = baud;
	comCfg.dcb.fBinary = 1;
	comCfg.dcb.fDtrControl = DTR_CONTROL_ENABLE;
	comCfg.dcb.fRtsControl = RTS_CONTROL_ENABLE;
	comCfg.dcb.ByteSize = 8;

	SetCommConfig(hSerial, &comCfg, sizeof(comCfg));

	// Code must specify a valid timeout value (0 means don't timeout)
	//	const DWORD ctoms = (timeout * 100);
	const DWORD ctoms = (1 * 100);
	COMMTIMEOUTS cto = { ctoms, 1, ctoms, 1, ctoms };
	SetCommTimeouts(hSerial, &cto);

	// Configure Windows to Monitor the serial device for Character Reception
	SetCommMask(hSerial, EV_RXCHAR);

	if (purge) {
		PurgeComm(hSerial, PURGE_RXABORT);
		PurgeComm(hSerial, PURGE_TXABORT);
		PurgeComm(hSerial, PURGE_RXCLEAR);
		PurgeComm(hSerial, PURGE_TXCLEAR);
	}

	return _open_osfhandle((intptr_t)hSerial, 0);
#else
	int fdDev = open(devpath, O_RDWR | O_CLOEXEC | O_NOCTTY);

	if (unlikely(fdDev == -1))
	{
		if (errno == EACCES)
			applog(LOG_ERR, "Do not have user privileges required to open %s", devpath);
		else
			applog(LOG_DEBUG, "Open %s failed, errno:%d", devpath, errno);

		return -1;
	}

	struct termios my_termios;

	tcgetattr(fdDev, &my_termios);

#ifdef TERMIOS_DEBUG
	termios_debug(devpath, &my_termios, "before");
#endif

	switch (baud) {
	case 0:
		break;
	case 19200:
		cfsetispeed(&my_termios, B19200);
		cfsetospeed(&my_termios, B19200);
		break;
	case 38400:
		cfsetispeed(&my_termios, B38400);
		cfsetospeed(&my_termios, B38400);
		break;
	case 57600:
		cfsetispeed(&my_termios, B57600);
		cfsetospeed(&my_termios, B57600);
		break;
	case 115200:
		cfsetispeed(&my_termios, B115200);
		cfsetospeed(&my_termios, B115200);
		break;
		// TODO: try some higher speeds with the Icarus and BFL to see
		// if they support them and if setting them makes any difference
		// N.B. B3000000 doesn't work on Icarus
	default:
		applog(LOG_WARNING, "Unrecognized baud rate: %lu", baud);
	}

	my_termios.c_cflag &= ~(CSIZE | PARENB);
	my_termios.c_cflag |= CS8;
	my_termios.c_cflag |= CREAD;
	my_termios.c_cflag |= CLOCAL;

	my_termios.c_iflag &= ~(IGNBRK | BRKINT | PARMRK |
		ISTRIP | INLCR | IGNCR | ICRNL | IXON);
	my_termios.c_oflag &= ~OPOST;
	my_termios.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);

	// Code must specify a valid timeout value (0 means don't timeout)
	my_termios.c_cc[VTIME] = (cc_t)timeout;
	my_termios.c_cc[VMIN] = 0;

#ifdef TERMIOS_DEBUG
	termios_debug(devpath, &my_termios, "settings");
#endif

	tcsetattr(fdDev, TCSANOW, &my_termios);

#ifdef TERMIOS_DEBUG
	tcgetattr(fdDev, &my_termios);
	termios_debug(devpath, &my_termios, "after");
#endif

	if (purge)
		tcflush(fdDev, TCIOFLUSH);
	return fdDev;
#endif
}

size_t _serial_read(int fd, char *buf, size_t bufsiz, char *eol)
{
	size_t len, tlen = 0;
	while (bufsiz) {
		len = _read(fd, buf, eol ? 1 : bufsiz);
		if (unlikely(len == -1))
			break;
		tlen += len;
		if (eol && *eol == buf[0])
			break;
		buf += len;
		bufsiz -= len;
	}
	return tlen;
}

int serial_recv(int fd, char *buf, size_t bufsize, size_t *readlen)
{
	BOOL Status;
	DWORD dwEventMask = 0;
	const HANDLE fh = (HANDLE)_get_osfhandle(fd);

	/*	Status = WaitCommEvent(fh, &dwEventMask, NULL); //Wait for the character to be received

	if (Status == FALSE) {
	printf("\n    Error! in Setting WaitCommEvent()");
	return(-1);
	}*/

	DWORD NoBytesRead = 0;
	char TempChar;
	int len = 0;

	do {
		Status = ReadFile(fh, &TempChar, sizeof(TempChar), &NoBytesRead, NULL);

		buf[len++] = TempChar;

		if (len == bufsize)
			break;

	} while (NoBytesRead > 0);

	*readlen = (size_t)len;
	return(0);
}

//add one fpga manually for now
static void fpga_detect(void)
{
	struct cgpu_info *cgpu;

	opt_g_threads = 1;

	opencl_drv.max_diff = 65536;

	cgpu = &gpus[0];
	cgpu->deven = DEV_ENABLED;
	cgpu->drv = &opencl_drv;
	cgpu->thr = NULL;
	cgpu->device_id = 0;
	cgpu->threads = opt_g_threads;
	cgpu->virtual_gpu = 0;
	cgpu->algorithm = default_profile.algorithm;
	add_cgpu(cgpu);
}

static void reinit_fpga_device(struct cgpu_info *gpu)
{
	tq_push(control_thr[gpur_thr_id].q, gpu);
}

static void get_fpga_statline_before(char *buf, size_t bufsiz, struct cgpu_info *gpu)
{
	float gt = 0.0f;

	tailsprintf(buf, bufsiz, "%5.1fC ", gt);
	tailsprintf(buf, bufsiz, "        ");
	tailsprintf(buf, bufsiz, "| ");
}

static void get_fpga_statline(char *buf, size_t bufsiz, struct cgpu_info *gpu)
{
	tailsprintf(buf, bufsiz, " I:%2d", gpu->intensity);
}

static bool fpga_thread_prepare(struct thr_info *thr)
{
	char name[256];
	struct timeval now;
	struct cgpu_info *cgpu = thr->cgpu;
	int gpu = cgpu->device_id;
	int virtual_gpu = cgpu->virtual_gpu;
	int i = thr->id;
	static bool failmessage = false;
	int buffersize = BUFFERSIZE;


	strcpy(name, "");
	applog(LOG_INFO, "Init FPGA thread %i FPGA %i virtual FPGA %i", i, gpu, virtual_gpu);

	if (!cgpu->name)
		cgpu->name = strdup("FPGA");

	applog(LOG_INFO, "initCl() finished. Found %s", name);
	cgtime(&now);
	get_datestamp(cgpu->init, sizeof(cgpu->init), &now);

	return true;
}

extern char devpath[512];
extern int devbaud;
extern int devtimeout;
extern int fd;

static bool fpga_thread_init(struct thr_info *thr)
{
	struct cgpu_info *gpu = thr->cgpu;
	int r;

	thr->cgpu_data = 0;// thrdata;
	gpu->status = LIFE_WELL;
	gpu->device_last_well = time(NULL);
	
	fd = serial_open(devpath, devbaud, devtimeout, 1);

	return fd == 0 ? false : true;
}

static bool fpga_prepare_work(struct thr_info __maybe_unused *thr, struct work *work)
{
	work->blk.work = work;
	if (work->pool->algorithm.precalc_hash)
		work->pool->algorithm.precalc_hash(&work->blk, 0, (uint32_t *)(work->data));
	thr->pool_no = work->pool->pool_no;

	return true;
}

#include "sph/sph_blake.h"

void bswap(unsigned char *b, int len)
{
	if ((len & 3) != 0) {
		printf("bswap error: len not multiple of 4\n");
		return;
	}

	while (len) {
		unsigned char t[4];

		t[0] = b[0];
		t[1] = b[1];
		t[2] = b[2];
		t[3] = b[3];
		b[0] = t[3];
		b[1] = t[2];
		b[2] = t[1];
		b[3] = t[0];
		b += 4;
		len -= 4;
	}
}

static void reverse(unsigned char *b, int len)
{
	static unsigned char bt[1024];
	int i, j;

	if (len > 128) {
		system("pause");
		exit(0);
	}
	//	bt = (unsigned char*)malloc(len + 1);

	for (i = 0, j = len; i < len;) {
		bt[i++] = b[--j];
	}

	memcpy(b, bt, len);

	//	free(bt);
}

static int64_t fpga_scanhash(struct thr_info *thr, struct work *work, int64_t __maybe_unused max_nonce)
{
	const int thr_id = thr->id;
	struct opencl_thread_data *thrdata = (struct opencl_thread_data *)thr->cgpu_data;
	struct cgpu_info *gpu = thr->cgpu;

	char *ob_hex;
	
	unsigned char sdata[80];
	unsigned char wbuf[56];
	unsigned char buf[56];
	unsigned int nonce;
	sph_blake256_context lyra2z_blake_mid;

	memset(wbuf, 0, 52);

	unsigned char input[(512 + 128) / 8] = {
		0x00, 0x00, 0x00, 0x20, 0x6E, 0x7E, 0x5F, 0xA2, 0x11, 0x0B, 0xA7, 0x79, 0xED, 0x8D, 0xD3, 0x4D,
		0x1F, 0xF7, 0x33, 0x21, 0x96, 0x79, 0xD3, 0x8E, 0x69, 0x0A, 0x58, 0xE1, 0xDE, 0x1D, 0x04, 0x00,
		0x00, 0x00, 0x00, 0x00, 0xE7, 0x7D, 0x10, 0x0C, 0x31, 0x9D, 0x75, 0x5B, 0xB9, 0x58, 0x13, 0xD9,
		0x79, 0xD7, 0x80, 0xD8, 0xBB, 0xAC, 0x20, 0x5A, 0xC7, 0x33, 0x36, 0xFC, 0xD9, 0x77, 0xEA, 0xD2,
		0x86, 0x09, 0xF6, 0xF1, 0x27, 0x08, 0x3F, 0x5B, 0x51, 0xC5, 0x19, 0x1B, 0x00, 0x02, 0xE2, 0xC0,

	};

	memcpy(sdata, work->data, 80);

	bswap(sdata, 80);

	sph_blake256_init(&lyra2z_blake_mid);
	sph_blake256(&lyra2z_blake_mid, sdata, 64);

	wbuf[48] = work->target[0x1F];
	wbuf[49] = work->target[0x1E];
	wbuf[50] = work->target[0x1D];
	wbuf[51] = work->target[0x1C];

	memcpy(wbuf + 0, &lyra2z_blake_mid.H[0], 32);
	memcpy(wbuf + 32, ((unsigned char*)(sdata)) + 64, 16);

//	bswap(wbuf, 32);
	reverse(wbuf, 44);
	bswap(wbuf, 12);

#define SERIAL_READ_SIZE 8

	struct FPGA_INFO {
		int device_fd;
		int timeout;
		double Hs;		// Seconds Per Hash
	};

	unsigned char ob_bin[44], nonce_buf[SERIAL_READ_SIZE];
	struct timeval tv_start, tv_finish, elapsed, tv_end, diff;
	int ret;

	struct cgpu_info *serial_fpga;
	struct FPGA_INFO *info;
	struct FPGA_INFO _info;

	serial_fpga = thr->cgpu;
	info = &_info;

	info->device_fd = fd;
	info->Hs = 200;
	info->timeout = 5;


	// Send Data To FPGA
	//	ret = write(fd, ob_bin, sizeof(ob_bin));
	_write(fd, wbuf, 52);

	/*	if (ret != sizeof(ob_bin)) {
	applog(LOG_ERR, "%s%i: Serial Send Error (ret=%d)", serial_fpga->drv->name, serial_fpga->device_id, ret);
	//serial_fpga_close(thr);
	dev_error(serial_fpga, REASON_DEV_COMMS_ERROR);
	return 0;
	}*/

	if (opt_debug) {
		char *ob_hex = bin2hex(wbuf, 52);
		applog(LOG_ERR, "Serial FPGA %d sent: %s", serial_fpga->device_id, ob_hex);
		free(ob_hex);
	}

	elapsed.tv_sec = 0;
	elapsed.tv_usec = 0;
	cgtime(&tv_start);

	size_t len;

	applog(LOG_DEBUG, "%s%i: Begin Scan For Nonces", serial_fpga->drv->name, serial_fpga->device_id);
	while (thr && !thr->work_restart) {

		memset(buf, 0, 8);

		// Check Serial Port For 1/10 Sec For Nonce  
		//		ret = read(fd, nonce_buf, SERIAL_READ_SIZE);
		ret = serial_recv(fd, (char*)buf, 8, &len);

		// Calculate Elapsed Time
		cgtime(&tv_end);
		timersub(&tv_end, &tv_start, &elapsed);

		if (ret == 0 && len != 8) {		// No Nonce Found
			if (elapsed.tv_sec > info->timeout) {
				applog(LOG_ERR, "%s%i: End Scan For Nonces - Time = %d sec", serial_fpga->drv->name, serial_fpga->device_id, elapsed.tv_sec);
				//thr->work_restart = true;
				break;
			}
			continue;
		}
		else if (ret != 0) { //(ret < SERIAL_READ_SIZE) {
			applog(LOG_ERR, "%s%i: Serial Read Error (ret=%d)", serial_fpga->drv->name, serial_fpga->device_id, ret);
			//serial_fpga_close(thr);
			dev_error(serial_fpga, REASON_DEV_COMMS_ERROR);
			break;
		}

		memcpy((char *)&nonce, buf, 4);

		nonce = swab32(nonce);

		//		curr_hw_errors = serial_fpga->hw_errors;

		applog(LOG_ERR, "%s%i: Nonce Found - %08X (%5.1fMhz)", serial_fpga->drv->name, serial_fpga->device_id, nonce, (double)(1 / (info->Hs * 1000000)));
		submit_nonce(thr, work, nonce);
		break;

		// Update Hashrate
		//		if (serial_fpga->hw_errors == curr_hw_errors)
		//			info->Hs = ((double)(elapsed.tv_sec) + ((double)(elapsed.tv_usec)) / ((double)1000000)) / (double)nonce;

	}


	int hash_count = 200;// ((double)(elapsed.tv_sec) + ((double)(elapsed.tv_usec)) / ((double)1000000)) / info->Hs;

						 //	free_work(work);
	return hash_count;


	/* The amount of work scanned can fluctuate when intensity changes
	* and since we do this one cycle behind, we increment the work more
	* than enough to prevent repeating work */
	work->blk.nonce += gpu->max_hashes;

	return 1000000/2300;
}

static void fpga_thread_shutdown(struct thr_info *thr)
{
	if(fd)
		_close(fd);
	fd = 0;
	thr->cgpu_data = NULL;
}

struct device_drv opencl_drv = {
  /*.drv_id = */            DRIVER_opencl,
  /*.dname = */             "fpga",
  /*.name = */              "FPG",
  /*.drv_detect = */        fpga_detect,
  /*.reinit_device = */     reinit_fpga_device,
  /*.get_statline_before =*/get_fpga_statline_before,
  /*.get_statline = */      get_fpga_statline,
  /*.api_data = */          NULL,
  /*.get_stats = */         NULL,
  /*.identify_device = */   NULL,
  /*.set_device = */        NULL,

  /*.thread_prepare = */    fpga_thread_prepare,
  /*.can_limit_work = */    NULL,
  /*.thread_init = */       fpga_thread_init,
  /*.prepare_work = */      fpga_prepare_work,
  /*.hash_work = */         NULL,
  /*.scanhash = */          fpga_scanhash,
  /*.scanwork = */          NULL,
  /*.queue_full = */        NULL,
  /*.flush_work = */        NULL,
  /*.update_work = */       NULL,
  /*.hw_error = */          NULL,
  /*.thread_shutdown = */   fpga_thread_shutdown,
  /*.thread_enable =*/      NULL,
  false,
  0,
  0
};
