#pragma once

#include <cstdlib>
#include <string>
#include <vector>
#include <set>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <boost/function.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "glog/logging.h"

namespace sjm {
/*!
  Just some utility functions I use a lot in my code
*/
namespace util {
// Returns true if filename exists
inline bool fexists(const char * const filename) {
  std::ifstream ifile(filename);
  return ifile;
}

inline void SplitStringUsing(
    const std::string& input, const std::string& separators,
    std::vector<std::string>* results) {
  results->clear();
  boost::split(*results, input, boost::is_any_of(separators));
}

// Expands a leading '~' in a path into a full path by replacing the
// '~' with the current user's complete path to their home
// folder. Does nothing to paths that don't have a leading '~'.
inline std::string expand_user(std::string path) {
  if (!path.empty() and path[0] == '~') {
    CHECK(path.size() == 1 || path[1] == '/') <<
        "Attempting to expand malformed path.";
    const char* home = getenv("HOME");
    if (home || (home = getenv("USERPROFILE"))) {
      // Replace the first character with the home directory.
      path.replace(0, 1, home);
    }
  }
  return path;
}

inline
void ReadFileToStringOrDie(const std::string& filename, std::string* dest) {
  dest->clear();
  FILE* f = fopen(sjm::util::expand_user(filename).c_str(), "rb");
  CHECK(f != NULL) << "Error opening " << filename;
  const size_t kBufferSize = 1024;
  char buffer[kBufferSize];
  size_t num_read = 0;
  while ((num_read = fread(buffer, sizeof(char), kBufferSize, f)) > 0) {
    dest->append(buffer, num_read);
  }
  CHECK_EQ(0, fclose(f));
}

inline
void ReadLinesFromFileIntoVectorOrDie(const std::string& filename,
                                      std::vector<std::string>* dest) {
  dest->clear();
  std::string data;
  ReadFileToStringOrDie(sjm::util::expand_user(filename), &data);
  std::vector<std::string> file_lines;
  boost::split(file_lines, data, boost::is_any_of("\n"));
  for (size_t i = 0; i < file_lines.size(); ++i) {
    if (!file_lines[i].empty()) {
      dest->push_back(file_lines[i]);
    }
  }
}

inline
void WriteStringToFileOrDie(const std::string& filename,
                            const std::string& source) {
  FILE* f = fopen(sjm::util::expand_user(filename).c_str(), "wb");
  CHECK(f != NULL) << "Error opening " << filename << " for writing.";
  CHECK_EQ(source.size(), fwrite(source.data(), 1, source.size(), f));
  CHECK_EQ(0, fclose(f));
}

inline
void AppendStringToFileOrDie(const std::string& filename,
                             const std::string& source) {
  FILE* f = fopen(sjm::util::expand_user(filename).c_str(), "a");
  CHECK(f != NULL) << "Error opening " << filename << " for append.";
  CHECK_EQ(source.size(), fwrite(source.data(), 1, source.size(), f));
  CHECK_EQ(0, fclose(f));
}
    /*!
      Checks whether or not a given extension is in a list of extensions
      
      @param extension the exension query
      @param extensionList the list of valid extensions
      @return bool returns true if the given extension is valid
    */
    inline bool extensionCheck(const std::string & extension, const std::set<std::string> & extensionList)
    {
      return extensionList.find(boost::algorithm::to_lower_copy(extension)) != extensionList.end();
    }

    /*!
      Applies a function that operates on a file to the given file.

      @param func the void function object that takes a file path as a parameter
      @param filePath a file path
    */
    inline void applyFunctionToFile(boost::function1<void, const boost::filesystem::path&> func, const boost::filesystem::path & filePath)
    {
      func(filePath);
    }

    /*!  Takes a function that operates on a file and applies it to
      any files in a given path hierarchy with valid extensions.

      @param func the void function object that takes a file path as a parameter
      @param filePath a file path
      @param validExtensions the set of extensions that func can operate on
      @param recursive flag to cause recursive call of this function with any directories found in filePath
    */
    inline void recursiveFunctionApplication(boost::function1<void, const boost::filesystem::path&> func,
					     const boost::filesystem::path & filePath,
					     const std::set<std::string> & validExtensions,
					     bool recursive)
    {
      if ( boost::filesystem::exists(filePath) )
	{
	  if ( boost::filesystem::is_directory(filePath) )
	    {
	      if ( recursive )
		{
		  boost::filesystem::directory_iterator dit(filePath);
		  boost::filesystem::directory_iterator end;

		  for ( ; dit != end; ++dit )
		    {
		      recursiveFunctionApplication(func, *dit, validExtensions, recursive);
		    }
		}
	    }
	  else if ( extensionCheck(boost::filesystem::extension(filePath), validExtensions) )
	    {
	      applyFunctionToFile(func, filePath);
	    } // end if-else directory/file check
	} // end if file exists
    } // end recursive function application


  } // end namespace util
} // end namespace sjm
